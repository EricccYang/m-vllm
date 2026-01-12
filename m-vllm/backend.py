import m_vllm_csrc





class MyAttBackEnd:
    def __init__(self):
        self.name = "my_backend"

    def FalshAttentionBackend(q, k, v, mask) -> torch.Tensor:
        return m_vllm_csrc.FalshAttention(q, k, v, mask)
