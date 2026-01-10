from .BaseModel import BaseModel



class Qwen3Model(BaseModel):
    def __init__(self, model_path= "qwen3-3b"):
        super().__init__(model_path)

    def test(a):
        print(a)