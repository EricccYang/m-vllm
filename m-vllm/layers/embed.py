import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from m_vllm.data_classes.context import get_context


class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vocab_size_p = vocab_size // self.tp_size
        self.start_vocab_idx = self.tp_rank * self.vocab_size_p
        self.end_vocab_idx = self.start_vocab_idx + self.vocab_size_p
        self.weight = nn.Parameter(torch.randn(self.vocab_size_p, hidden_size))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param : nn.Parameter, weight: torch.Tensor):
        pa = param.data
        weight = weight.narrow(0, self.tp_rank * self.vocab_size_p, self.vocab_size_p)
        pa.copy_(weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            mask = (input_ids >= self.start_vocab_idx ) & ( input_ids <= self.end_vocab_idx)
            input_ids = mask * ( input_ids - self.start_vocab_idx)
        y = F.embedding(input_ids, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
    




class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        bias : bool = False
    ):
        assert not bias
        super().__init__(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            input_ids = input_ids[last_indices].contiguous()
        logits = F.linear(input_ids, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits







