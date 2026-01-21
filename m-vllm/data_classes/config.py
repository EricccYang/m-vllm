import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class TpGroupConfig:
    id: int
    group_size: int
    world_rank: int



@dataclass
class ModelConfig:
    model_path: str
    hf_config: AutoConfig | None = None



    # def __init__(self, model_path: str):
    #     self.model_path = model_path
    #     self.mode_config = AutoConfig.from_pretrained(model_path)


    # def get_model_path(self):
    #     return self.model_path

    # def set_model_path(self, model_path: str):
    #     self.model_path = model_path

    # def get_model_config(self):
    #     return self.model_config




@dataclass
class GlobalConfig:
    model_name: str
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    gpu_memory_utilization: float = 0.9
    hf_config: AutoConfig | None = None
    eos :int  = -1
    tp_config : TpGroupConfig = TpGroupConfig()
    model_config : ModelConfig = ModelConfig()




