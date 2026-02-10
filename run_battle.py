import random
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
import os
import json
from werewolf.agents import agent_registry
import yaml
import shutil

def eval(env, agent_list, roles_):
    for agent in agent_list:
        agent.reset()
    done = False
    obs = env.reset(roles=roles_)
    while not done:
        current_act_idx = obs['current_act_idx']
        action = agent_list[current_act_idx - 1].act(obs)
        obs, reward, done, info = env.step(action)
    if done:
        if info['Werewolf'] == 1:
            return 'Werewolf win'
        elif info['Werewolf'] == -1:
            return 'Villager win'
    return "Draw" 

def get_replaced_wolf_id(replace_players, assgined_roles):
    replace_type = replace_players.split("_")[1]
    if replace_type == "last":
        reversed_lst = assgined_roles[::-1]
        index_in_reversed = reversed_lst.index("Werewolf")
        replace_id = len(assgined_roles) - 1 - index_in_reversed
    elif replace_type == "random":
        indexes = [i for i, x in enumerate(assgined_roles) if x == "Werewolf"]
        replace_id = random.choice(indexes)
    else:
        raise NotImplementedError
    return replace_id

def get_replaced_simple_villager_ids(assgined_roles, replace_number):
    indexes = [i for i, x in enumerate(assgined_roles) if x == "Villager"]
    replace_ids = random.sample(indexes, replace_number)
    return replace_ids

def get_replaced_villager_ids(assgined_roles, replace_number):
    indexes = [i for i, x in enumerate(assgined_roles) if x != "Werewolf"]
    replace_ids = random.sample(indexes, replace_number)
    return replace_ids

def assign_agents_and_roles(assgined_roles, all_agent_models, env_param, agent_config, log_dir):
    agent_list = []
    
    def create_agent(role_type, idx, params, log_path):
        return agent_registry.build_agent(role_type, idx, params, env_param, log_path)

    if "replace" not in agent_config:
        for i, role in enumerate(assgined_roles):
            log_file = os.path.join(log_dir, f"Player_{i+1}.jsonl")
            
            key = role.lower()
            if key == "werewolf":
                type, agent_param = all_agent_models["werewolf"]
            elif key in all_agent_models:
                type, agent_param = all_agent_models[key]
            else:
                type, agent_param = all_agent_models["villager"]
                
            agent = create_agent(type, i, agent_param, log_file)
            agent_list.append(agent)
        return agent_list
    
    replace_players = agent_config["replace"]["replace_player"]
    replace_role = replace_players.split("_")[0]
    
    if replace_role == "werewolf":
        repalce_id = get_replaced_wolf_id(replace_players, assgined_roles)
        for i, role in enumerate(assgined_roles):
            log_file = os.path.join(log_dir, f"Player_{i+1}.jsonl")
            if role.lower() == "werewolf" and i != repalce_id:
                type, agent_param = all_agent_models["werewolf"]
            elif role.lower() == "werewolf" and i == repalce_id:
                type, agent_param = all_agent_models["replace"]
            else:
                type, agent_param = all_agent_models["villager"]
            agent = create_agent(type, i, agent_param, log_file)
            agent_list.append(agent)
        return agent_list
        
    else:
        print("Warning: Complex replace logic detected. Using standard assignment for stability.")
        for i, role in enumerate(assgined_roles):
            log_file = os.path.join(log_dir, f"Player_{i+1}.jsonl")
            if role.lower() == "werewolf":
                type, agent_param = all_agent_models["werewolf"]
            else:
                type, agent_param = all_agent_models["villager"]
            agent = create_agent(type, i, agent_param, log_file)
            agent_list.append(agent)
        return agent_list


def define_agents(agent_config, env_config, args, assgined_roles, log_dir):
    env_param = {
        "n_player": env_config["n_player"],
        "n_role": env_config["n_role"]
    }
    all_agent_models = {} 
    for group in agent_config.keys():
        if group == "replace": continue 
        agent_config[group]["model_params"].update(env_param)
        model_type = agent_config[group]["model_type"]
        if model_type not in [i[0] for g,i in all_agent_models.items()]:
            all_agent_models[group] = agent_registry.build(model_type, **agent_config[group]["model_params"])
        else:
            for g, i in all_agent_models.items():
                if model_type == i[0]:
                    all_agent_models[group] = model_type, i[1]
                    break
    return assign_agents_and_roles(assgined_roles, all_agent_models, env_param, agent_config, log_dir)


def check_agent_config(agent_config):
    if "sft" in agent_config["werewolf"]["model_type"].lower() or "makto" in agent_config["werewolf"]["model_type"].lower():
        assert agent_config["werewolf"]["model_params"].get("port", None) is not None, f'No port provided'
    if "sft" in agent_config["villager"]["model_type"].lower() or "makto" in agent_config["villager"]["model_type"].lower():
        assert agent_config["villager"]["model_params"].get("port", None) is not None, f'No port provided'


def update_config_based_on_rank(agent_config, rank):
    if rank == 0:
        return agent_config
    print(f"Applying Rank {rank} offset to ports...")
    for role_name, config in agent_config.items():
        if role_name == "replace": continue
        if "model_params" in config and "port" in config["model_params"]:
            old_port = config["model_params"]["port"]
            new_port = old_port + rank
            config["model_params"]["port"] = new_port
            print(f"  [{role_name}] Port updated: {old_port} -> {new_port}")
    return agent_config


def main_cli(args):
    base_log_path = args.log_save_path
    os.makedirs(base_log_path, exist_ok=True)
    
    parsed_yaml = yaml.safe_load(open(args.config))
    agent_config = parsed_yaml["agent_config"]
    env_config = parsed_yaml["env_config"]
    
    agent_config = update_config_based_on_rank(agent_config, args.rank)

    if not os.path.exists(os.path.join(os.path.dirname(base_log_path), "config.yaml")):
        try:
            with open(os.path.join(os.path.dirname(base_log_path), "config.yaml"), "w") as f:
                yaml.dump(parsed_yaml, f)
        except Exception as e:
            pass

    check_agent_config(agent_config)
    
    for i in range(args.num_games):
        game_id = i + 1
        print(f"\n[GPU {args.rank}] >>> Starting Game {game_id}/{args.num_games} <<<")
        
        current_game_dir = os.path.join(base_log_path, f"game_{game_id}")
        os.makedirs(current_game_dir, exist_ok=True)
        
        roles = ["Werewolf"] * env_config["n_werewolf"] + ["Villager"] * env_config["n_villager"] + \
                ["Seer"] * env_config["n_seer"] + ["Witch"] * env_config["n_witch"] + \
                ["Guard"] * env_config["n_guard"] + ["Hunter"] * env_config["n_hunter"]
        random.shuffle(roles)
        print(f"Game {game_id} Roles: {roles}")
        
        meta_info = {
            "game_id": game_id,
            "roles": roles,
            "role_map": {f"Player_{idx+1}": role for idx, role in enumerate(roles)},
            "rank": args.rank,
            "timestamp": time.time()
        }
        
        env = WerewolfTextEnvV0(**env_config)
        agent_list = define_agents(agent_config, env_config, args, roles, log_dir=current_game_dir)
        
        begin = time.time()
        try:
            result = eval(env, agent_list, roles)
        except Exception as e:
            print(f"Game {game_id} Error: {e}")
            result = "Error"
            import traceback
            traceback.print_exc()
        
        duration = time.time() - begin
        print(f"Game {game_id} Finished. Result: {result}. Duration: {duration:.2f}s")
        
        meta_info["result"] = result
        meta_info["duration"] = duration
        
        with open(os.path.join(current_game_dir, "meta_info.json"), "w", encoding='utf-8') as f:
            json.dump(meta_info, f, indent=4, ensure_ascii=False)
            
        print(f"Data saved to: {current_game_dir}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="configs/gpt4_vs_gpt4.yaml", help="path to config")
    argparser.add_argument('--log_save_path', type=str, default=None)
    
    argparser.add_argument('--rank', type=int, default=0, help="GPU rank (0 or 1)")
    argparser.add_argument('--num_games', type=int, default=1, help="Total games to run on this GPU")
    
    args = argparser.parse_args()
    main_cli(args)