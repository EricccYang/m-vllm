import os
from dataclasses import dataclass
from transformers import AutoConfig


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