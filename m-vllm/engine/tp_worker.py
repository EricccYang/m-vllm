import torch
import torch.distributed as dist
import pickle
import multiprocessing as mp
import logging
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from multiprocessing import Value

from m_vllm.engine.model_runner import ModelRunner
from m_vllm.engine.scheduler import Scheduler
from m_vllm.data_classes.batch import RunBatch, Sequence
from m_vllm.data_classes.config import GlobalConfig, TpGroupConfig, ModelConfig
from m_vllm.data_classes.context import set_context

logger = logging.getLogger(__name__)


def launch_tp_group(
    group_rank: int,
    model_config: ModelConfig,
    global_config: GlobalConfig,
    engine_post_req_event: Event,
    engine_get_result_event: Event,
    model_ready: Value,
):
    # 在子进程中重新配置日志（spawn 模式不会继承父进程的日志配置）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Python 3.8+ 支持，强制重新配置
    )
    logger.info("launching tp group")
    worker = HeadTPWorker(
        group_rank,
        model_config,
        global_config,
        engine_post_req_event,
        engine_get_result_event,
        model_ready,
    )
    worker.start()


# 几类参数：tp group的信息， model的信息，还有超参数, 每一类写成一个结构体
class TPWorker:
    def __init__(
        self,
        tp_group_config: TpGroupConfig,
        model_config: ModelConfig,
        global_config: GlobalConfig,
        event: Event = None,
    ):
        logger.info(
            f"TPWorker init: rank {tp_group_config.group_rank}, group size {tp_group_config.group_size}, world size {global_config.world_size}, inter_rank {tp_group_config.group_inter_rank}, world rank {tp_group_config.group_inter_rank + tp_group_config.group_rank * global_config.pipeline_parallel_size}"
        )
        self.tp_group_config = tp_group_config
        self.model_config = model_config
        self.global_config = global_config

        self.world_size = global_config.world_size
        self.world_rank = (
            tp_group_config.group_inter_rank
            + tp_group_config.group_rank * global_config.pipeline_parallel_size
        )
        self.event = event
        if not self.tp_group_config.is_head() and self.event is not None:
            self.shm = SharedMemory(name="inter_tp_group", create=False)


    
    def init_dist_and_load(self):
        dist.init_process_group(
            "nccl",
            "tcp://localhost:2333",
            world_size=self.world_size,
            rank=self.world_rank,
        )
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_config.qwen3_config.dtype)
        torch.cuda.set_device(self.world_rank)
        torch.set_default_device("cuda")
        self.model_runner = ModelRunner(self.model_config.model_path, self.global_config)
        self.allocate_kv_cache()
        torch.set_default_dtype(default_dtype)

    # 这里pp可能有问题，需要修改
    def allocate_kv_cache(self):
        logger.info("Starting KV cache allocation")
        gpu_memory_utilization = self.global_config.gpu_memory_utilization
        free, total = torch.cuda.mem_get_info()
        # logger.info(f"free {free} bytes, total {total} bytes")
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # logger.info(f"peak {peak} bytes, current {current} bytes")
        cf = self.global_config.model_config.qwen3_config
        if cf is None:
            logger.error("qwen3_config is None!")
            raise ValueError("qwen3_config is None, cannot allocate KV cache")
        logger.info(
            f"config: num_hidden_layers={cf.num_hidden_layers}, num_attention_heads={cf.num_attention_heads}, head_dim={cf.head_dim}, hidden_size={cf.hidden_size}"
        )
        block_bytes = (
            2
            * cf.num_hidden_layers
            * cf.num_attention_heads
            * cf.head_dim
            * cf.dtype.itemsize
            * self.global_config.kvcache_block_size
        )
        # logger.info(f"block_bytes: {block_bytes}")
        available_memory = total * gpu_memory_utilization - used - (peak - current)
        # logger.info(f"available_memory: {available_memory} bytes")
        
        # self.global_config.num_kvcache_blocks = int(available_memory // block_bytes)
        self.global_config.num_kvcache_blocks = 16
        # logger.info(f"allocate {self.global_config.num_kvcache_blocks} kvcache blocks")
        if self.global_config.num_kvcache_blocks <= 0:
            logger.error(
                f"Failed to allocate KV cache: num_kvcache_blocks={self.global_config.num_kvcache_blocks}, available_memory={available_memory}, block_bytes={block_bytes}"
            )
        assert self.global_config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            cf.num_hidden_layers,
            self.global_config.num_kvcache_blocks,
            self.global_config.kvcache_block_size,
            cf.num_key_value_heads,
            cf.head_dim,
        ).cuda(non_blocking=True)
        layer_id = 0
        for module in self.model_runner.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def read_from_tp_group_mem(self):
        logger.info("world rank %d is reading from tp group mem, world size %d", self.world_rank, self.world_size)
        assert self.world_size > 1 and self.world_rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        run_batch = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return run_batch

    def prepare_batch(self, run_batch: RunBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """在 HeadTPWorker 中实现，这里只是一个占位"""
        raise NotImplementedError("Should be implemented in HeadTPWorker")

    def _loop(self):
        logger.info("TPWorker number %d loop started", self.world_rank)
        while True:
            run_batch = self.read_from_tp_group_mem()
            input_ids, positions = self.prepare_batch(run_batch)
            self.model_runner.run(input_ids, positions)


class HeadTPWorker(TPWorker):
    """TP 组的头节点，负责与 Engine 通信、调度和协调其他 TP Workers"""

    def __init__(
        self,
        group_rank: int,
        model_config: ModelConfig,
        global_config: GlobalConfig,
        engine_post_req_event: Event,
        engine_get_result_event: Event,
        model_ready: Value,
    ):
        self.engine_post_req_event = engine_post_req_event
        self.engine_get_result_event = engine_get_result_event
        self.model_ready = model_ready
        self.get_req_shm = SharedMemory(name="mem_engine_to_tp_head", create=False)
        self.send_result_shm = SharedMemory(name="mem_tp_head_to_engine", create=False)

        self.tp_group_config = TpGroupConfig(
            group_rank=group_rank,
            group_size=global_config.tensor_parallel_size,
            group_inter_rank=0,
        )
        if self.tp_group_config.group_size > 1:
            self.tp_group_shm = SharedMemory(
                name="inter_tp_group", create=True, size=2**20
            )
            self.tp_worker_events = []
            self.tp_group_processes = []

    
        super().__init__(self.tp_group_config, model_config, global_config, event=None)

        logger.info("tp group config in head tp worker: %s", self.tp_group_config)
        self._launch_tp_workers()
        logger.info("Initializing distributed environment and loading model...")
        self.init_dist_and_load()
        logger.info("Model loaded successfully")
        self.scheduler = Scheduler(global_config)
        logger.info("Scheduler created")
        
        # 通知父进程模型已加载完成
        with self.model_ready.get_lock():
            self.model_ready.value = 1
        logger.info("Model ready flag set")

    def start(self):
        self._loop()

    def _launch_tp_workers(self):
        
        ctx = mp.get_context("spawn")
        # 使用相同的 spawn 上下文创建 Manager
        manager = ctx.Manager()

        for i in range(1, self.tp_group_config.group_size):
            event = manager.Event()
            worker_tp_config = TpGroupConfig(
                group_rank=self.tp_group_config.group_rank,
                group_size=self.tp_group_config.group_size,
                group_inter_rank=self.tp_group_config.group_inter_rank + i,
            )
            process = ctx.Process(
                target=self._launch_worker,
                args=(worker_tp_config, self.model_config, self.global_config, event),
            )
            process.start()
            self.tp_group_processes.append(process)
            self.tp_worker_events.append(event)

    @staticmethod
    def _launch_worker(
        tp_group_config: TpGroupConfig,
        model_config: ModelConfig,
        global_config: GlobalConfig,
        event: Event,
    ):
        # 在子进程中重新配置日志（spawn 模式不会继承父进程的日志配置）
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,  # Python 3.8+ 支持，强制重新配置
        )
        worker = TPWorker(tp_group_config, model_config, global_config, event)
        worker.init_dist_and_load()
        worker._loop()

    def schedule(self):
        return self.scheduler.schedule()

    def read_request_from_engine(self):
        self.engine_post_req_event.wait()
        n = int.from_bytes(self.get_req_shm.buf[0:4], "little")
        request = pickle.loads(self.get_req_shm.buf[4 : n + 4])
        self.engine_post_req_event.clear()
        return request

    def send_result_to_engine(self, results):
        data = pickle.dumps(results)
        n = len(data)
        if n + 4 > 2**20:
            raise ValueError(f"Result too large: {n} bytes")
        self.send_result_shm.buf[0:4] = n.to_bytes(4, "little")
        self.send_result_shm.buf[4 : n + 4] = data
        self.engine_get_result_event.set()

    def send_batch_to_tp_workers(self, batch):
        if self.world_size <= 1:
            return
        data = pickle.dumps(batch)
        n = len(data)
        self.tp_group_shm.buf[0:4] = n.to_bytes(4, "little")
        self.tp_group_shm.buf[4 : n + 4] = data
        for event in self.tp_worker_events:
            event.set()

    def prepare_batch(self, run_batch: RunBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if run_batch.prefill_mode:
            return self.prepare_prefill(run_batch)
        else:
            return self.prepare_decode(run_batch)

    def prepare_prefill(self, run_batch: RunBatch):
        input_ids = []
        positions = []
        cu_seq_q = [0]
        cu_seq_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = []
        
        logger.info(f"prepare_prefill: num_sequences={len(run_batch.sequences)}")
        if not run_batch.sequences:
            logger.error("prepare_prefill: run_batch.sequences is empty!")
            raise ValueError("Cannot prepare prefill batch with empty sequences")
            
        for seq in run_batch.sequences:
            seq_len = len(seq.token_ids)
            # 在 prefill 模式下，处理所有 tokens，不管 cache 状态
            # num_cached_tokens 在 allocate 时可能被更新，但 prefill 应该处理所有 tokens
            real_len = seq_len
            logger.info(f"Prefill seq: seq_len={seq_len}, num_cached_tokens={seq.num_cached_tokens}, "
                       f"real_len={real_len}, token_ids={seq.token_ids[:10] if len(seq.token_ids) > 10 else seq.token_ids}")
            
            # 确保 real_len > 0
            if real_len <= 0:
                logger.warning(f"Sequence has real_len={real_len}, skipping. seq_len={seq_len}")
                continue
                
            # token_ids 是 list，在 prefill 模式下处理所有 tokens
            input_ids.extend(seq.token_ids)
            positions.extend(list[int](range(seq_len)))
            seqlen_q = real_len
            seqlen_k = seq_len
            cu_seq_q.append(cu_seq_q[-1] + real_len)
            cu_seq_k.append(cu_seq_k[-1] + real_len)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * seq.block_size
                if i == seq.num_blocks - 1:
                    # 最后一个 block 可能不完整，使用实际的 token 数量
                    end = start + seq.last_block_num_tokens
                else:
                    # 完整的 block，使用 block_size
                    end = start + seq.block_size
                slot_mapping.extend(list[int](range(start, end)))
        
        # 检查 input_ids 是否为空
        if not input_ids:
            logger.error("prepare_prefill: input_ids is empty!")
            raise ValueError("Cannot prepare prefill batch with empty input_ids")
            
        if cu_seq_q[-1] < cu_seq_k[-1]:
            block_tables = self.prepare_block_tables(run_batch.sequences)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seq_q = torch.tensor(cu_seq_q, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seq_k = torch.tensor(cu_seq_k, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seq_q,
            cu_seq_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, run_batch: RunBatch):
        input_ids = []
        positions = []
        slot_mapping = []
        block_tables = []
        context_lens = []
        for seq in run_batch.sequences:
            # last_token 是 list 的最后一个元素
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * seq.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(run_batch.sequences)
        set_context(True, context_lens, slot_mapping, block_tables)
        return input_ids, positions

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def _loop(self):
        logger.info("HeadTPWorker loop started")
        while True:
            request = self.read_request_from_engine()
            self.scheduler.add_request(request)

            run_batch = self.scheduler.schedule()
            # 如果没有可调度的序列，跳过本次循环
            if run_batch is None:
                continue
            input_ids, positions = self.prepare_batch(run_batch)

            if self.world_size > 1:
                self.send_batch_to_tp_workers(run_batch)

            logits = self.model_runner.run(input_ids, positions)
            token_ids = self.model_runner.sample(logits, run_batch)
            self.scheduler.post_process(run_batch.sequences, token_ids)
            self.send_result_to_engine(run_batch.sequences)
