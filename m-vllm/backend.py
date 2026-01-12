from .kernels.m_vllm_csrc import rms_norm_kernel
import torch




class MyAttBackEnd:
    def __init__(self):
        self.name = "my_backend"

    def RMSNormBackend(self,q, k, v, mask) -> torch.Tensor:
        return rms_norm_kernel(q)
