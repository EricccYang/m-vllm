from re import M
from .BaseModel import BaseModel
from m_vllm_csrc import add
from backend import MyAttBackEnd



class Qwen3Model(BaseModel):
    def __init__(self, model_path= "qwen3-3b"):
        self.attension = MyAttBackEnd
        super().__init__(model_path)

    def test(a):
        return add(1, 2)


    def forward(q,k,v,batch):
        self.attension.FalshAttentionBackend(q,k,v,batch)