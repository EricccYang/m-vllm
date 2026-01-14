# import m_vllm.data_classes.Batch as Batch
import torch
from m_vllm.backend import GatedMlpBackend




class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.impl = LinearBackend()

    def forward(self, x: torch.Tensor):
        return self.impl.forward(x,self.weight,self.bias)
