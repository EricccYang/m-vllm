import torch
from m_vllm.models.Qwen3 import Qwen3ForCausalLM
from m_vllm.models.modeloader import load_model
from m_vllm.data_classes.ModelConfig import ModelConfig
from transformers import AutoConfig
from m_vllm.data_classes.config import Qwen3Config





if __name__ == "__main__":

    path = "./qwen3-14B"
    hf_config = AutoConfig.from_pretrained(path)
    
    model_config = Qwen3Config.from_hf_config(hf_config)
    model = Qwen3ForCausalLM(model_config)
    load_model(model, path)

    
    print(model(torch.randn(1, 1024)))
    print(model.test())
