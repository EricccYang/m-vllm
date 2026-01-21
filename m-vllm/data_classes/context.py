from dataclasses import dataclass




@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q: list[int]
    cu_seqlens_k: list[int]
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: list[int]
    block_tables: list[int]


_CONTEXT = Context(False, [], [], 0, 0, [], [])

def get_context():
    return _CONTEXT

def reset_context():
    global _CONTEXT
    _CONTEXT = Context(False, [], [], 0, 0, [], [])

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables)