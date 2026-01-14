import torch
from m_vllm.models.Qwen3 import Qwen3Model
from m_vllm.data_classes.ModelConfig import ModelConfig
from transformers import AutoConfig




if __name__ == "__main__":

    path = "./qwen3-14"
    hf_config = AutoConfig.from_pretrained(path)
    
    
    model = Qwen3Model("qwen3-3b", hf_config)
    load_model(model, config.model)

    
    print(model(torch.randn(1, 1024)))
    print(model.test())
