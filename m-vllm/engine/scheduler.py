from typing import Deque
import logging
from m_vllm.data_classes.request import Request
from m_vllm.data_classes.batch import RunBatch, Sequence
from m_vllm.engine.paged_kvcache import BlockManager
from m_vllm.data_classes.config import GlobalConfig
from m_vllm.data_classes.batch import SequenceStatus

logger = logging.getLogger(__name__)


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
        sequence = Sequence(request.token_ids, request.sampling_params)
        self.waiting_queue.append(sequence)
        return sequence

    def is_finished(self):
        return not self.waiting_queue and not self.running_queue

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq.block_table)
        seq.status = SequenceStatus.WAITING
        self.waiting_queue.appendleft(seq)

    def schedule(self) -> RunBatch | None:
        sqs = list[Sequence]()
        num_seqs = 0
        num_batched_tokens = 0

        logger.info(f"Schedule called: waiting_queue={len(self.waiting_queue)}, running_queue={len(self.running_queue)}, "
                   f"free_blocks={len(self.block_manager.free_blocks)}, max_num_seqs={self.max_num_seqs}")

        if not self.waiting_queue and not self.running_queue:
            return None

        # prefill: 优先处理等待队列中的新请求
        while self.waiting_queue and num_seqs < self.max_num_seqs:
            seq = self.waiting_queue[0]
            seq_len = len(seq)
            seq_blocks = seq.num_blocks
            can_allocate = self.block_manager.can_allocate(seq_blocks)
            tokens_check = len(seq) + num_batched_tokens > self.max_num_seqs
            
            logger.debug(f"Prefill check: seq_len={seq_len}, seq_blocks={seq_blocks}, "
                        f"num_batched_tokens={num_batched_tokens}, can_allocate={can_allocate}, "
                        f"tokens_check={tokens_check}")
            
            if tokens_check or not can_allocate:
                logger.info(f"Prefill break: tokens_check={tokens_check}, can_allocate={can_allocate}, "
                           f"seq_len={seq_len}, num_batched_tokens={num_batched_tokens}, "
                           f"free_blocks={len(self.block_manager.free_blocks)}")
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting_queue.popleft()
            self.running_queue.append(seq)
            sqs.append(seq)
            logger.debug(f"Prefill added seq: seq_len={len(seq)}, num_seqs={num_seqs}, sqs_size={len(sqs)}")
        
        if sqs:
            logger.info(f"Returning prefill batch: {len(sqs)} sequences")
            return RunBatch(sqs, prefill_mode=True)

        # decode: 处理正在运行的序列
        logger.info(f"Starting decode phase: running_queue={len(self.running_queue)}")
        while self.running_queue and num_seqs < self.max_num_seqs:
            seq = self.running_queue.popleft()
            seq_len = len(seq)
            can_append = self.block_manager.can_append(seq)
            
            logger.debug(f"Decode check: seq_len={seq_len}, can_append={can_append}, "
                        f"free_blocks={len(self.block_manager.free_blocks)}, "
                        f"seq_len % block_size={seq_len % self.block_manager.block_size}")
            
            while not can_append:
                # 抢占
                logger.info(f"Cannot append seq (len={seq_len}), attempting preemption. "
                           f"running_queue={len(self.running_queue)}, free_blocks={len(self.block_manager.free_blocks)}")
                if self.running_queue:
                    preempted = self.running_queue.pop()
                    logger.info(f"Preempting seq (len={len(preempted)})")
                    self.preempt(preempted)
                    can_append = self.block_manager.can_append(seq)
                    logger.debug(f"After preemption: can_append={can_append}, free_blocks={len(self.block_manager.free_blocks)}")
                else:
                    logger.info(f"No other seqs to preempt, preempting current seq (len={seq_len})")
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                sqs.append(seq)
                logger.debug(f"Decode added seq: seq_len={len(seq)}, num_seqs={num_seqs}, sqs_size={len(sqs)}")
        
        logger.warning(f"Schedule result: sqs_size={len(sqs)}, waiting_queue={len(self.waiting_queue)}, "
                      f"running_queue={len(self.running_queue)}, free_blocks={len(self.block_manager.free_blocks)}, "
                      f"num_seqs={num_seqs}, num_batched_tokens={num_batched_tokens}")
        
        if not sqs:
            logger.error(f"No sequences scheduled! waiting_queue={len(self.waiting_queue)}, "
                         f"running_queue={len(self.running_queue)}, free_blocks={len(self.block_manager.free_blocks)}")
        
        assert sqs
        self.running_queue.extendleft(reversed(sqs))
        return RunBatch(sqs, prefill_mode=False)

    def post_process(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):  # type: ignore
            # 将新的 token_id 追加到 list 中
            if hasattr(token_id, 'item'):
                token_id = token_id.item()
            else:
                token_id = int(token_id)
            seq.token_ids.append(token_id)
            logger.info(f"Post process: seq.token_ids={seq.token_ids}, token_id={token_id}")
            seq.num_tokens += 1
            logger.info( f"eos = {self.eos}, seq.ignore_eos = {seq.ignore_eos}, token_id = {token_id}")
            if not seq.ignore_eos and token_id == self.eos:
                seq.status = SequenceStatus.FINISHED
            if seq.max_tokens and len(seq) >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
            if seq.is_finished:
                self.running_queue.remove(seq)
                self.waiting_queue.append(seq)
