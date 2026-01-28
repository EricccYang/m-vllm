import logging
import torch
from m_vllm.models.Qwen3 import Qwen3ForCausalLM
from m_vllm.models.modeloader import load_model
from m_vllm.engine.engine import LLMEngine
from m_vllm.data_classes.server_args import ServerArgs
from transformers import AutoConfig
from m_vllm.data_classes.request import Request
import time
# Qwen3Config 是 PretrainedConfig 的子类，如果需要特定类型可以使用
# from transformers import Qwen3Config

# 配置日志系统（只需在程序入口配置一次）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == "__main__":
    path = "../Qwen3-4B/"
    # hf_config = AutoConfig.from_pretrained(path)

    # model_config = Qwen3Config(hf_config)
    # model = Qwen3ForCausalLM(model_config)
    # load_model(model, path)
    args = ServerArgs(path, tensor_parallel_size=1, pipeline_parallel_size=1)
    engine = LLMEngine(args)

    engine.add_request(Request(input_str="Hello, world!"))
    engine.add_request(Request(input_str="Hello, world! 2"))
    engine.start_running()


    while True:
        time.sleep(1)


    # print(model(torch.randn(1, 1024)))
    # print(model.test())
