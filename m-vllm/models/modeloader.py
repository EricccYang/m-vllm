import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open



def default_weight_loader(model_part: nn.Parameter, load_weight: torch.Tensor):
    model_part.data.copy_(load_weight)
    
    

def load_model(model: nn.Module, model_path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for file_weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in key:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = file_weight_name.replace(k, v)
                        model_part = model.get_parameter(param_name)
                        loader = getattr(model_part, "weight_loader")
                        loader(model_part, f.get_tensor(file_weight_name), shard_id)
                        break
                else:
                    model_part = model.get_parameter(file_weight_name)
                    loader = getattr(model_part,"weight_loader", default_weight_loader)
                    loader(model_part, f.get_tensor(file_weight_name))


                








