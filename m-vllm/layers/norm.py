import torch
from torch import nn
from typing import Optional, Tuple
# from m_vllm.backend import MyAttBackEnd


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        # self.impl = RMSNormBackend()

    @torch.compile
    def add_forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        type = x.dtype
        x = x.to(torch.float32)
        x.add_(residual.to(torch.float32))
        residual = x.to(type)  # 所以x是32位算的
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(type).mul_(self.weight)
        return x, residual

    @torch.compile
    def forward_without_residual(self, x: torch.Tensor) -> torch.Tensor:
        origin_type = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(origin_type).mul_(self.weight)
        return x

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            return self.add_forward(x, residual)
        else:
            return self.forward_without_residual(x)
