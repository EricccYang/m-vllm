import torch
from .models.Qwen3 import Qwen3Model


if __name__ == "__main__":
    model = Qwen3Model("qwen3-3b")
    print(model(torch.randn(1, 1024)))
    print(model.test())
