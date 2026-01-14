from re import M
from transformers import Qwen3Config

from m_vllm.models.BaseModel import BaseModel
from m_vllm.kernels.m_vllm_csrc import add
from m_vllm.backend import MHAAttBackEnd
from m_vllm.data_classes.ModelConfig import ModelConfig 
from m_vllm.data_classes.Batch import Batch
from m_vllm.layers.norm import RMSNorm
from m_vllm.layers.GatedMlp import GatedMLP
from m_vllm.layers.rope import RotaryPositionEmbedding
from m_vllm.layers.linear import Linear
from m_vllm.backend import GatedMlpBackend



class GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.impl = GatedMlpBackend()
        self.down_proj = Linear()
        self.gate_up_proj = Linear()

    def forward(self, x: torch.Tensor):
        return self.impl.forward(x)



# attention 
class Qwen3Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, max_position_embeddings: int, base: int = 10000):
        self.name = "attention"
        self.qkv_proj = Linear()
        self.o_proj = Linear()
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()


        self.mha_att_backend = MHAAttBackEnd()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rotary_emb = RotaryPositionEmbedding(num_heads, head_dim, max_position_embeddings, base)




    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        #统一@
        qkv = self.qkv_proj(x)
        
        #分开
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        #reshape
        q = q.view(-1, self.num_heads, self.head_dim)   #[ batch_size*seq_len, num_heads, head_dim]
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        #qwen3特有 q k norm, 单头norm
        #  k_norm.weight  [128]
        #  q_norm.weight   [128]
        q = self.q_norm(q)
        k = self.k_norm(k)

        #旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        #attension
        h = self.mha_att_backend.forward(q, k, v, mask)

        #o_proj
        h = self.o_proj(h)

        return h



class Qwen3DecoderLayer(nn.Module):
    def __init__(self, model_config: Qwen3Config):
        super().__init__()
        self.model_config = model_config
        self.input_layernorm = RMSNorm()
        self.post_attention_layernorm = RMSNorm()
        self.selt_attn = Qwen3Attention(
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            max_position_embeddings=model_config.max_position_embeddings,
            base=model_config.base
        )
        self.mlp= GatedMLP()


    def output_proj(self, x: torch.Tensor):
        return x

    def forward(self, batch: Batch):
        #
        h = self.input_layernorm(batch)
        h = self.selt_attn(h, h, h, batch.mask)
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return h



class Qwen3Model(BaseModel):
    def __init__(self, model_path= "qwen3-3b", model_config: Qwen3Config = None):
        super().__init__()
        self.model_config = model_config
        self.layers = nn.ModuleList([Qwen3DecoderLayer(model_config) for _ in range(model_config.num_layers)])
        

    def test(a):
        return add(1, 2)


    def forward(self,batch: Batch):
        for layer in self.layers:
            h = layer(batch)
        return h




# 模型加载需要符合 safetensors 格式, 模型类及层类的成员变量命名需要符合 safetensors的内部定义
class Qwen3CausalLM(BaseModel):
    packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    }
    def __init__(self, model_path= "qwen3-3b", model_config: Qwen3Config = None):
        super().__init__()
        self.model= Qwen3Model(model_path, model_config)   

    def forward(self,batch: Batch):
        self.model.forward(batch)