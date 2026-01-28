from re import M
from transformers import Qwen3Config
import torch
from torch import nn
from typing import Optional
from m_vllm.models.BaseModel import BaseModel
from m_vllm.kernels.m_vllm_csrc import add
from m_vllm.backends.backend import MHAAttBackEnd
from m_vllm.layers.norm import RMSNorm
from m_vllm.layers.mlp import MLP
from m_vllm.layers.rope import get_rope
from m_vllm.layers.linear import (
    Linear,
    QKVParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from m_vllm.layers.activation import SiluAndMul
from m_vllm.layers.embed import VocabParallelEmbedding, ParallelLMHead
from m_vllm.layers.attension import Attention

import logging
logger = logging.getLogger(__name__)



class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # self.impl = GatedMlpBackend()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, intermediate_size, bias=False
        )
        self.down_proj = RowParallelLinear(hidden_size, hidden_size, bias=False)
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor):
        up = self.gate_up_proj(x)
        x = self.act(up)
        x = self.down_proj(x)
        return x


# attention
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: int = 10000,
        norm_eps: float = 1e-5,
        qkv_bias: bool = False,
        hidden_size: int = 128,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.name = "attention"
        self.norm_eps = norm_eps
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.q_size = head_dim * num_heads
        self.kv_size = head_dim * num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=qkv_bias,
        )

        self.q_norm = RMSNorm(head_dim, norm_eps)
        self.k_norm = RMSNorm(head_dim, norm_eps)
        self.rotary_emb = get_rope(head_dim, max_position_embeddings, self.rope_theta)

        self.mha_att_backend = Attention(num_heads, head_dim, self.scaling, num_kv_heads)

        # 参数需要再弄明白一点
        # o_proj 的输入大小应该是 num_heads * head_dim，而不是 hidden_size
        # 因为 flash_attn 返回的输出形状是 [batch*seq_len, num_heads, head_dim]
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        # 统一@
        qkv = self.qkv_proj(x)

        # 分开
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # reshape
        q = q.view(
            -1, self.num_heads, self.head_dim
        )  # [ batch_size*seq_len, num_heads, head_dim]
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # qwen3特有 q k norm, 单头norm
        #  k_norm.weight  [128]
        #  q_norm.weight   [128]
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        # attension
        o = self.mha_att_backend(q, k, v)
        logger.info(f"Qwen3Attention: o.shape={o.shape}, num_heads={self.num_heads}, head_dim={self.head_dim}, hidden_size={self.hidden_size}")
        logger.info(f"  o.flatten(1,-1).shape={o.flatten(1, -1).shape}, expected hidden_size={self.hidden_size}")

        # o_proj
        output = self.o_proj(o.flatten(1, -1))

        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, model_config: Qwen3Config):
        super().__init__()
        self.model_config = model_config
        self.input_layernorm = RMSNorm(
            model_config.hidden_size, model_config.rms_norm_eps
        )
        self.selt_attn = Qwen3Attention(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_attention_heads,
            head_dim=model_config.head_dim,
            num_kv_heads=model_config.num_key_value_heads,
            max_position_embeddings=model_config.max_position_embeddings,
            norm_eps=model_config.rms_norm_eps,
            qkv_bias=model_config.attention_bias,
            rope_theta=getattr(model_config, "rope_theta", 1000000),
            rope_scaling=getattr(model_config, "rope_scaling", None),
        )
        self.post_attention_layernorm = RMSNorm(
            model_config.hidden_size, model_config.rms_norm_eps
        )
        self.mlp = MLP(model_config.hidden_size, model_config.intermediate_size)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.selt_attn(hidden_states, position_ids)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(BaseModel):
    def __init__(self, model_config: Qwen3Config = None):
        super().__init__()
        self.model_config = model_config
        logger.info(f"num_hidden_layers: {model_config.num_hidden_layers}")
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(model_config) for _ in range(model_config.num_hidden_layers)]
        )
        self.embed_tokens = VocabParallelEmbedding(
            model_config.vocab_size, model_config.hidden_size
        )
        self.norm = RMSNorm(model_config.hidden_size, model_config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states)
        hidden_states, _ = self.norm(hidden_states)
        return hidden_states


# 模型加载需要符合 safetensors 格式, 模型类及层类的成员变量命名需要符合 safetensors的内部定义
class Qwen3ForCausalLM(BaseModel):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, model_config: Qwen3Config = None):
        super().__init__()
        self.model = Qwen3Model(model_config)
        self.lm_head = ParallelLMHead(model_config.vocab_size, model_config.hidden_size)

    def forward(self, positions: torch.Tensor, input_ids: torch.Tensor):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
