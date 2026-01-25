from typing import Any

import pickle
from sre_parse import Tokenizer
import torch.multiprocessing as mp
from m_vllm.engine.tp_worker import launch_tp_group
from m_vllm.data_classes.config import TpGroupConfig
from m_vllm.data_classes.server_args import ServerArgs
from queue import Queue
from transformers import AutoConfig, AutoTokenizer
from m_vllm.data_classes.config import ModelConfig
from transformers.configuration_utils import PretrainedConfig
from m_vllm.models.Qwen3 import Qwen3Config
from m_vllm.data_classes.request import Request
from multiprocessing.shared_memory import SharedMemory
from m_vllm.data_classes.config import GlobalConfig, TpGroupConfig
# from m_vllm.utils.logger import init_logger
import logging
import atexit

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def _cleanup_shared_memory(name: str):
    """清理已存在的共享内存"""
    try:
        shm = SharedMemory(name=name, create=False)
        shm.close()
        shm.unlink()
        logger.info(f"Cleaned up existing shared memory: {name}")
    except FileNotFoundError:
        # 共享内存不存在，这是正常情况
        pass
    except Exception as e:
        logger.warning(f"Failed to cleanup shared memory {name}: {e}")


class LLMEngine:
    def __init__(self, args: ServerArgs):
        # 请求设置参数
        self.args = args
        logger.info(f"Server args: {self.args}")
        
        # 从文件加载模型配置和 tokenizer
        
        self.model_config = ModelConfig(model_path=args.model_path, qwen3_config=AutoConfig.from_pretrained(args.model_path))
        logger.info(f"Model config: {self.model_config}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        logger.info("loaded tokenizer")

        # tp组处理 todo pp等其他组
        self.tp_group_size = self.args.pipeline_parallel_size
        
        # launch tp组的领头worker，tp组内worker由领头worker启动
        self.global_config = GlobalConfig(model_name=args.model_path, model_config=self.model_config, eos=self.tokenizer.eos_token_id, tensor_parallel_size=self.args.tensor_parallel_size, pipeline_parallel_size=self.args.pipeline_parallel_size)
        self.init_worker_group()

        #  api请求队列
        self.req_queue = Queue()
        

    def init_worker_group(self):
        ctx = mp.get_context("spawn")
        manager = mp.Manager()
        
        # 清理可能存在的旧共享内存
        _cleanup_shared_memory("mem_engine_to_tp_head")
        _cleanup_shared_memory("mem_tp_head_to_engine")
        _cleanup_shared_memory("inter_tp_group")
        
        self.post_req_shm = SharedMemory(name="mem_engine_to_tp_head", create=True, size=2**20)
        self.get_result_shm = SharedMemory(name="mem_tp_head_to_engine", create=True, size=2**20)
        self.post_req_event = manager.Event()  # 通知 HeadTPWorker 请求已写入
        self.get_result_event = manager.Event()  # 通知 Engine 结果已写入
        
        # 注册清理函数，确保程序退出时释放共享内存
        atexit.register(self.cleanup)

        self.processes = []
        process = ctx.Process(
            target=launch_tp_group, 
            args=(0, self.model_config, self.global_config, self.post_req_event, self.get_result_event)
        )
        process.start()
        self.processes.append(process)


        # 管理 batch 的能力应该交给谁？
    # 当前设计：Engine 负责接收请求，HeadTPWorker 负责调度和 batching
    def start_running(self):
        """Engine 主循环"""
        while True:
            # 1. 从队列获取请求
            request = self.req_queue.get()
            
            if request.input_str is None:
                continue
            else:
                logger.info("Engine received request: %s", request.input_str)
            # 2. Tokenize（在 CPU 上执行）

            request.token_ids = self.tokenizer.encode(request.input_str)
            
            # 3. 发送给 HeadTPWorker
            self.send_to_worker(request)
            
            # 4. 接收结果（同步等待）
            results = self.get_result_from_worker()
            
            # 5. 处理结果
            self.process_results(results)


    def send_to_worker(self, request: Request):
        data = pickle.dumps(request)
        n = len(data)
        if n + 4 > 2**20:
            raise ValueError(f"Request too large: {n} bytes, max {2**20 - 4}")
        
        self.post_req_shm.buf[0:4] = n.to_bytes(4, "little")
        self.post_req_shm.buf[4:n+4] = data    
        self.post_req_event.set()

    def get_result_from_worker(self):
        self.get_result_event.wait()
        n = int.from_bytes(self.get_result_shm.buf[0:4], "little")
        data = pickle.loads(self.get_result_shm.buf[4:n+4])
        self.get_result_event.clear()
        return data

 



    def add_request(self, request: Request):
        self.req_queue.put(request)

    def reply_request(self, request: Request):
        request.reply()

    def detokenize(self, request: Request):
        tk = self.tokenizer.decode(request.token_ids)
        return "str"

    def process_results(self, results: list[Request]):
        """处理返回的结果"""
        for req in results:
            text = self.detokenize(req)
            # 发送结果给用户
            if hasattr(req, 'callback'):
                req.callback(text)

    def cleanup(self):
        """清理资源，释放共享内存"""
        logger.info("Cleaning up LLMEngine resources...")
        try:
            if hasattr(self, 'post_req_shm'):
                self.post_req_shm.close()
                self.post_req_shm.unlink()
                logger.info("Cleaned up post_req_shm")
        except Exception as e:
            logger.warning(f"Error cleaning up post_req_shm: {e}")
        
        try:
            if hasattr(self, 'get_result_shm'):
                self.get_result_shm.close()
                self.get_result_shm.unlink()
                logger.info("Cleaned up get_result_shm")
        except Exception as e:
            logger.warning(f"Error cleaning up get_result_shm: {e}")
        
        # 清理子进程
        try:
            if hasattr(self, 'processes'):
                for process in self.processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                logger.info("Cleaned up processes")
        except Exception as e:
            logger.warning(f"Error cleaning up processes: {e}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构函数中的异常




            

        


