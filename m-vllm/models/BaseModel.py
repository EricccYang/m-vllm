import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor((1,1))

