from re import M
from .BaseModel import BaseModel
from ..kernels.m_vllm_csrc import add
from ..backend import MyAttBackEnd



class Qwen3Model(BaseModel):
    def __init__(self, model_path= "qwen3-3b"):
        self.attension = MyAttBackEnd()
        super().__init__(model_path)

    def test(a):
        return add(1, 2)


    def forward(self,q):
        k = q
        v = q
        batch = 1
        self.attension.RMSNormBackend(q,k,v,batch)
        return q