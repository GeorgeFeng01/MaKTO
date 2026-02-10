import os
from typing import Dict
import openai
from pydantic import BaseModel


class Registry(BaseModel):
    """Registry for storing and building classes."""

    name: str
    entries: Dict = {}
    translator_entries: Dict = {} 

    def register(self, keys: list):
        def decorator(cls):
            for key in keys:
                if key in self.entries:
                    raise ValueError(f"Key {key} is already registered with a different class.")
                self.entries[key] = cls
            return cls
        return decorator


    def build(self, type: str, **kwargs):
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        agent_params = {}

        if any(keyword in type.lower() for keyword in ["sft", "makto", "gpt", "o1"]):
            
            port = kwargs.get("port", 8000) 
            ip = kwargs.get("ip", None)
            
            if ip is None:
                base_url = f"http://localhost:{port}/v1"
            else:
                base_url = f"http://{ip}:{port}/v1"
            print(f"[Registry Check] Model '{type}' is connecting to: {base_url}")
            client = openai.OpenAI(
                api_key="EMPTY",
                base_url=base_url,
            )

            if "gpt" in type.lower() or "o1" in type.lower():
                llm_name = kwargs.get("llm", "qwen-8b-werewolf")
            else:
                llm_name = type

            agent_params = {
                "client": client,
                "tokenizer": None,
                "llm": llm_name,
                "temperature": kwargs.get("temperature", 0.7)
            }

        elif 'human' in type.lower():
            agent_params = {
                "client": None,
                "tokenizer": None,
                "llm": None,
                "temperature": 0
            }
            
        return type, agent_params

    def build_agent(self, type: str,
                    player_idx,
                    agent_param,
                    env_param,
                    log_file):
        
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        return self.entries[type](client=agent_param["client"],
                                    tokenizer=agent_param["tokenizer"],
                                    llm=agent_param["llm"],
                                    temperature=agent_param["temperature"],
                                    log_file=log_file)

    def get_all_entries(self):
        return self.entries