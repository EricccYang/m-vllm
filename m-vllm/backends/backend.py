import re
from m_vllm.kernels.m_vllm_csrc import rms_norm_kernel
import torch
import triton
import triton.language as tl


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
        self.k_cache = None
        self.v_cache = None

    def forward(self,q, k, v, mask) -> torch.Tensor:
        return rms_norm_kernel(q)



@triton.jit
def store_kv_cache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D : tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets  = ids * key_stride + tl.arrange(0,D)
    value_offsets = ids * value_stride + tl.arrange(0,D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offset = slot * D + tl.arange(0,D)
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)
    


def store_kv_cache(k_cache : torch.Tensor, v_cache : torch.Tensor, k : torch.Tensor, v : torch.Tensor, slot_mapping : torch.Tensor):
    N ,num_kv_heads, head_dim = k_cache.shape
    D = head_dim * num_kv_heads
    assert k.stride(-1) == 1 & v.stride(-1) == 1
    assert k.stride(1) == head_dim & v.stride(1) == head_dim
    assert k_cache.stride(1) == D & v_cache.stride(1) == D
    assert slot_mapping.numel() == N  #seq_len
    store_kv_cache_kernel[(N,)](key, key.stride(0), v, v.stride(0), k_cache, v_cache, slot_mapping, D)




class TritonAttensionBackend:
    def __init__(self):
        self.name = "trition_attension_backend"


    def forward(self, q, k, v, mask, k_cache, v_cache) -> torch.Tensor:
        context = get_context()
        store_kv_cache(k_cache, v_cache, k, v)
        if context.is_prefill:  
            o = prefill_kernel(q, k, v, mask, k_cache, v_cache)
        else:
            o = decode_kernel(q, k, v, mask, k_cache, v_cache)
        return o
