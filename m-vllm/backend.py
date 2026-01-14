from m_vllm.kernels.m_vllm_csrc import rms_norm_kernel
import torch




class RMSNormBackend:
    def __init__(self):
        self.name = "rms_norm_backend"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_kernel(x)


class GatedMlpBackend:
    def __init__(self):
        self.name = "gated_mlp_backend"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MHAAttBackEnd:
    def __init__(self):
        self.name = "my_backend"

    def forward(self,q, k, v, mask) -> torch.Tensor:
        return rms_norm_kernel(q)
