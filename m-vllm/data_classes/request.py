from typing import Optional
from itertools import count
from m_vllm.data_classes.batch import SamplingParams


class Request:
    counter = count()
    def __init__(self, input_str :str ):
        self.input_str = input_str
        self.req_ids = next(Request.counter)
        self.token_ids = []  # 使用 list 而不是 tensor
        self.sampling_params = SamplingParams(ignore_eos=0, max_tokens=0)
        self.answer : Optional[str] = None