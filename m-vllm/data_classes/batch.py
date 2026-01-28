from typing import TYPE_CHECKING
from enum import Enum, auto
from dataclasses import dataclass

if TYPE_CHECKING:
    from m_vllm.engine.paged_kvcache import Block

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    COMPLETED = auto()



@dataclass
class SamplingParams:
    ignore_eos: int
    max_tokens: int


class Sequence:
    def __init__(self, input_ids: list[int], sampling_params):
        self.num_tokens = len(input_ids)  # 使用 len 而不是 shape[0]
        self.token_ids = input_ids
        self.positions = list[int]()
        self.block_table = []
        self.block_size = 256
        self.num_cached_tokens = 0
        self.status = SequenceStatus.WAITING
        self.ignore_eos = sampling_params.ignore_eos
        self.max_tokens = sampling_params.max_tokens

    def __len__(self):
        return len(self.token_ids)

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]


    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def last_token(self):
        return self.token_ids[-1]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size


class RunBatch:
    def __init__(self, sequences: list[Sequence]):
        self.input_ids = None
        self.positions = None
        self.prefill_mode = True
        self.sequences = sequences  # 使用传入的 sequences 参数
