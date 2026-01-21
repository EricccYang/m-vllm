from typing import Deque, Sequence
from m_vllm.data_classes.request import Request
from m_vllm.data_classes.batch import RunBatch, Sequence
from m_vllm.engine.paged_kvcache import BlockManager
from m_vllm.data_classes.config import GlobalConfig
from m_vllm.data_classes.batch import SequenceStatus


class Scheduler:
    def __init__(self, config: GlobalConfig):
        self.max_num_seqs = 64
        self.running_queue = Deque[Sequence]()
        self.waiting_queue = Deque[Sequence]()
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.eos = config.eos

    def add_request(self, request: Request):
        sequence = Sequence(request.token_ids, None, request.sampling_params)
        self.waiting_queue.append(sequence)
        return sequence

    def is_finished(self):
        return not self.waiting_queue and not self.running_queue

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq.block_table)
        seq.status = SequenceStatus.WAITING
        self.waiting_queue.appendleft(seq)

    def schedule(self) -> RunBatch:
        # rb=  RunBatch()
        # assert self.waiting_queue
        sqs = list[Sequence]()
        num_seqs = 0
        num_batched_tokens = 0

        # prefill
        while self.waiting_queue and num_seqs < self.max_num_seqs:
            seq = self.waiting_queue[0]
            if (
                len(seq) + num_batched_tokens > self.max_num_seqs
                or not self.block_manager.can_allocate(len(seq))
            ):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting_queue.popleft()
            self.running_queue.append(seq)
            sqs.append(seq)
        if sqs:
            return RunBatch(sqs)

        # decode
        while self.running_queue and num_seqs < self.max_num_seqs:
            seq = self.running_queue.popleft()
            while not self.block_manager.can_append(seq):
                # 抢占
                if self.running_queue:
                    self.preempt(self.running_queue.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                sqs.append(seq)
        assert sqs
        self.running_queue.extendleft(reversed(sqs))
        return RunBatch(sqs)

    def post_process(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip[tuple[Sequence, int]](seqs, token_ids):
            seq.token_ids.extend(token_id)
            if not seq.ignore_eos and token_id == self.eos:
                seq.status = SequenceStatus.FINISHED
            if seq.max_tokens and len(seq) >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
            if seq.is_finished:
                self.running_queue.remove(seq)
                self.waiting_queue.append(seq)
