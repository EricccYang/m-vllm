
from re import I
from textwrap import wrap
import torch

from torch._subclasses.fake_impls import wordaround_stride_incorrect_op
from m_vllm.engine.model_runner import ModelRunner
from m_vllm.engine.scheduler import Scheduler
from m_vllm.data_classes.batch import RunBatch, Sequence
from m_vllm.data_classes.server_args import ServerArgs
import torch.distributed as dist
import pickle
from multiprocessing.shared_memory import SharedMemory
from m_vllm.data_classes.config import GlobalConfig
from m_vllm.data_classes.context import set_context

import multiprocessing as mp
from multiprocessing.synchronize import Event



#pp的时候这个tp要下一个pp组的头交互的，所以不能由engine来管理，要由tp_worker来管理



def launch_tp_group(config: ServerArgs, rank: int, post_req_event: Event, get_result_event: Event, global_config: GlobalConfig):
    """启动 TP 组的 Head Worker"""
    HeadTPWorker(config, rank, post_req_event, get_result_event, global_config)
    
    

class TPWorker:
    def __init__(self, world_size: int, world_rank: int, event: Event, global_config: GlobalConfig):

        self.world_size = world_size
        self.world_rank = world_rank
        self.event = event
        self.shm = mp.SharedMemory(create=True, size=1024)
        self.model_runner = ModelRunner(global_config.model_config.model_path, global_config)
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=world_rank)
        self.shm = SharedMemory(name="inter_tp_group", create=False) 
        self.allocate_kv_cache(global_config)
        self.loop()


    def allocate_kv_cache(self, global_config: GlobalConfig):
        gpu_memory_utilization = global_config.gpu_memory_utilization
        used, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        cf = global_config.hf_config
        block_bytes = ( 2 * cf.num_hidden_layers * cf.num_attention_heads * cf.hidden_size * cf.torch_dtype.itemsize * global_config.kvcache_block_size )
        global_config.num_kvcache_blocks = int( (total * gpu_memory_utilization - used - (peak - current)) // block_bytes )
        assert global_config.num_kvcache_blocks > 0
        #kv其实隔的挺远的这样， 是不是最好的方法不知道
        self.kv_cache = torch.empty(2, cf.num_hidden_layers, global_config.num_kvcache_blocks, global_config.kvcache_block_size, cf.num_attention_heads, cf.hidden_size)
        layer_id = 0
        for module in self.model_runner.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1


    def read_shm(self):
        assert self.world_size > 1 and self.world_rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args


    def read_from_tp_group_mem(self):
        assert self.world_size > 1 and self.world_rank > 0
        n = int.from_bytes(self.shm.buf[0:4], "little")
        return pickle.loads(self.shm.buf[4:n+4])

    def loop(self):
        while True:
            run_batch = self.read_from_tp_group_mem()
            self.model_runner.run(run_batch)  #no return





class HeadTPWorker(TPWorker):
    """TP 组的头节点，负责与 Engine 通信、调度和协调其他 TP Workers"""
    
    def __init__(self, config: ServerArgs, rank: int, post_req_event: Event, get_result_event: Event, global_config: GlobalConfig):
        
        self.tp_group_config = global_config.tp_config
        self.config = config
        self.world_size = self.tp_group_config.group_size
        self.world_rank = self.tp_group_config.world_rank
        
        # 事件：与 Engine 通信
        self.post_req_event = post_req_event  # Engine 通知请求就绪
        self.get_result_event = get_result_event  # 通知 Engine 结果就绪
        
        # 初始化调度器和模型
        self.scheduler = Scheduler(global_config)
        
        # 初始化分布式
        dist.init_process_group(
            "nccl", 
            "tcp://localhost:2333", 
            world_size=self.world_size, 
            rank=rank
        )
        
        # 共享内存：Engine -> HeadTPWorker
        self.get_req_shm = SharedMemory(name="mem_engine_to_tp_head", create=False)
        
        # 共享内存：HeadTPWorker -> Engine
        self.send_result_shm = SharedMemory(name="mem_tp_head_to_engine", create=False)
        
        # 共享内存：HeadTPWorker -> TP Workers（组内通信）
        if self.world_size > 1:
            self.tp_group_shm = SharedMemory(name="inter_tp_group", create=True, size=2**20)
            self.tp_worker_events = []
        
        # 启动其他 TP Workers
        self._launch_tp_workers()
        
        # 启动主循环
        self.loop()


    
    def _launch_tp_workers(self, global_config: GlobalConfig):
        """启动 TP 组内的其他 workers"""
        if self.world_size <= 1:
            return
        
        ctx = mp.get_context("spawn")
        for i in range(1, self.world_size):
            event = ctx.Event()
            process = ctx.Process(
                target=self._launch_worker,
                args=(self.config, i, event, global_config)
            )
            process.start()
            self.tp_group_config.processes.append(process)
            self.tp_worker_events.append(event)
    
    @staticmethod
    def _launch_worker(config, rank, event, tp_group_config):
        """静态方法：启动单个 TP Worker"""
        TPWorker(tp_group_config.group_size, rank, event)


    def schedule(self):
        return self.scheduler.schedule()


    def read_request_from_engine(self):
        """从 Engine 读取请求"""
        # 等待 Engine 通知 
        self.post_req_event.wait()
        
        # 读取数据
        n = int.from_bytes(self.get_req_shm.buf[0:4], "little")
        request = pickle.loads(self.get_req_shm.buf[4:n+4])
        
        # 清除事件
        self.post_req_event.clear()
        
        # 添加到调度器
        self.scheduler.add_request(request)
        return request

    def send_result_to_engine(self, results):
        """将结果发送回 Engine"""
        data = pickle.dumps(results)
        n = len(data)
        
        # 检查大小
        if n + 4 > 2**20:
            raise ValueError(f"Result too large: {n} bytes")
        
        # 写入共享内存
        self.send_result_shm.buf[0:4] = n.to_bytes(4, "little")
        self.send_result_shm.buf[4:n+4] = data
        
        # 通知 Engine
        self.get_result_event.set()

    def send_batch_to_tp_workers(self, batch):
        """将 batch 发送给其他 TP Workers"""
        if self.world_size <= 1:
            return
        
        data = pickle.dumps(batch)
        n = len(data)
        
        # 写入共享内存
        self.tp_group_shm.buf[0:4] = n.to_bytes(4, "little")
        self.tp_group_shm.buf[4:n+4] = data
        
        # 通知所有 workers
        for event in self.tp_worker_events:
            event.set()

    def prepare_batch(self, run_batch: RunBatch)->tuple[torch.Tensor, torch.Tensor]:
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
        for seq in run_batch.sequences:
            seq_len = len(seq.token_ids)
            real_len = seq_len - seq.num_cached_tokens
            input_ids.extend(seq.token_ids[seq.num_cached_tokens:])
            positions.extend(list[int](range(seq.num_cached_tokens,len)))
            seqlen_q = real_len
            seqlen_k = seq_len
            cu_seq_q.append(cu_seq_q[-1] + real_len)
            cu_seq_k.append(cu_seq_k[-1] + real_len)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * seq.block_size
                if i == seq.num_blocks - 1:
                    end = start + seq.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list[int](range(start, end)))
        if cu_seq_q[-1] < cu_seq_k[-1]:
            block_tables = self.prepare_block_tables(run_batch.sequences)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seq_q = torch.tensor(cu_seq_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seq_k = torch.tensor(cu_seq_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seq_q, cu_seq_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables)
        return input_ids, positions


    def prepare_decode(self, run_batch: RunBatch):
        input_ids = []
        positions = []
        slot_mapping = []
        block_tables = []
        context_lens = []
        for seq in run_batch.sequences:
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * seq.block_size + seq.last_block_num_tokens - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(run_batch.sequences)
        set_context(True, context_lens, slot_mapping, block_tables)
        return input_ids, positions
        

            
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables


    def loop(self):
        """Head Worker 主循环"""
        while True:
            # 1. 从 Engine 读取请求
            request = self.read_request_from_engine()
            self.scheduler.add_sequence(request)
            
            # 2. 调度（batching）
            run_batch = self.scheduler.schedule()

            input_ids, positions = self.prepare_batch(run_batch)
            
            # 3. 如果有多个 TP Workers，分发任务
            if self.world_size > 1:
                self.send_batch_to_tp_workers(input_ids, positions)
            
            # 4. Head Worker 自己执行推理
            logits = self.model_runner.run( input_ids, positions)
            
            # 5. 采样生成 token
            token_ids = self.model_runner.sample(logits, run_batch)
            
            # 6. 更新请求的 token_ids
            self.scheduler.post_process(run_batch.sequences, token_ids)
            
            # 7. 发送结果回 Engine
            self.send_result_to_engine(run_batch.requests)
            #todo
            #先记一下吧，回头再来修改算了