from functools import lru_cache
import torch
from torch import nn



def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).to(x.dtype)



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, rope_base = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_base = rope_base
        inv_freq = 1/(self.rope_base**(torch.arange(0, self.head_dim, 2) / self.head_dim))
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat([cos, sin], dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        return q, k




@lru_cache(maxsize=1)
def get_rope(head_dim, max_position_embeddings=2048, rope_theta=10000):
    rope = RotaryPositionEmbedding(head_dim, max_position_embeddings, rope_theta)
    return rope

