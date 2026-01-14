import torch
from m_vllm.backend import MyAttBackEnd


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.impl = RMSNormBackend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl.forward(x, self.weight)