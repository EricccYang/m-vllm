import os
from dataclasses import dataclass, field
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


@dataclass
class TpGroupConfig:
    group_rank: int
    group_size: int
    group_inter_rank: int

    def is_head(self):
        return self.group_inter_rank == 0


@dataclass
class ModelConfig:
    model_path: str
    qwen3_config: PretrainedConfig | None = None
    hf_config: PretrainedConfig | None = None

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
    eos: int = -1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    world_size: int = 1
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(model_path="", qwen3_config=None)
    )
