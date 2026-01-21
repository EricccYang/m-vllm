from typing import Any

import pickle
from sre_parse import Tokenizer
import torch.multiprocessing as mp
from m_vllm.engine.tp_worker import launch_tp_group, TpGroupConfig
from m_vllm.data_classes.server_args import ServerArgs
from queue import Queue
from transformers import AutoConfig, AutoTokenizer
from m_vllm.data_classes.request import Request
from multiprocessing.shared_memory import SharedMemory
from m_vllm.data_classes.config import GlobalConfig


class LLMEngine:
    def __init__(self, args: ServerArgs):
        self.args = args
        self.processes = []
        ctx = mp.get_context("spawn")

        # 加载模型配置和 tokenizer
        model_config = AutoConfig.from_pretrained(args.model, use_fast=True)    
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        
        # 创建共享内存：Engine -> HeadTPWorker (发送请求)
        self.post_req_shm = SharedMemory(name="mem_engine_to_tp_head", create=True, size=2**20)
        
        # 创建共享内存：HeadTPWorker -> Engine (接收结果)
        self.get_result_shm = SharedMemory(name="mem_tp_head_to_engine", create=True, size=2**20)
        
        # 创建同步事件
        self.post_req_event = ctx.Event()  # 通知 HeadTPWorker 请求已写入
        self.get_result_event = ctx.Event()  # 通知 Engine 结果已写入
        
        global_config = GlobalConfig(eos=self.tokenizer.eos_token_id, tp_config=TpGroupConfig(0, args.tensor_parallel_size, 0))

        # 启动 TP 组
        process = ctx.Process(
            target=launch_tp_group, 
            args=(args, 0, self.post_req_event, self.get_result_event, global_config)
        )
        process.start()
        self.processes.append(process)
        
        # 请求队列
        self.req_queue = Queue()
        

 
    def add_request(self, request: Request):
        self.req_queue.put(request)

    def reply_request(self, request: Request):
        request.reply()

    def detokenize(self, request: Request):
        tk = self.tokenizer.decode(request.token_ids)
        request.channel.send(tk)

    def send_to_worker(self, request: Request):
        """将请求发送给 HeadTPWorker"""
        data = pickle.dumps(request)
        n = len(data)
        
        # 检查数据大小
        if n + 4 > 2**20:
            raise ValueError(f"Request too large: {n} bytes, max {2**20 - 4}")
        
        # 写入共享内存
        self.post_req_shm.buf[0:4] = n.to_bytes(4, "little")
        self.post_req_shm.buf[4:n+4] = data
        
        # 通知 HeadTPWorker
        self.post_req_event.set()

    def get_result_from_worker(self):
        """从 HeadTPWorker 接收结果"""
        # 等待结果就绪
        self.get_result_event.wait()
        
        # 从结果共享内存读取
        n = int.from_bytes(self.get_result_shm.buf[0:4], "little")
        data = pickle.loads(self.get_result_shm.buf[4:n+4])
        
        # 清除事件
        self.get_result_event.clear()
        return data

    def detokenize(self, request: Request):
        """将 token IDs 转换为文本"""
        return self.tokenizer.decode(request.token_ids)
        

    def process_results(self, results: list[Request]):
        """处理返回的结果"""
        for req in results:
            text = self.detokenize(req)
            # 发送结果给用户
            if hasattr(req, 'callback'):
                req.callback(text)


    # 管理 batch 的能力应该交给谁？
    # 当前设计：Engine 负责接收请求，HeadTPWorker 负责调度和 batching
    def start(self):
        """Engine 主循环"""
        while True:
            # 1. 从队列获取请求
            request = self.req_queue.get()
            
            # 2. Tokenize（在 CPU 上执行）
            request.token_ids = self.tokenizer.encode(request.input_str)
            
            # 3. 发送给 HeadTPWorker
            self.send_to_worker(request)
            
            # 4. 接收结果（同步等待）
            results = self.get_result_from_worker()
            
            # 5. 处理结果
            self.process_results(results)

            

        


