import torch

from m_vllm.models.Qwen3 import Qwen3ForCausalLM
from transformers import Qwen3Config
from m_vllm.data_classes.batch import RunBatch
from m_vllm.layers.sampler import Sampler
from m_vllm.data_classes.config import GlobalConfig


# def launch_runner():
#     pass


class ModelRunner:
    def __init__(self, path: str, config: GlobalConfig):
        self.model = Qwen3ForCausalLM(config.model_config.qwen3_config)
        self.model.eval()  # 设置为评估模式
        self.sampler = Sampler()

    def run(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # 禁用梯度计算
            h = self.model(input_ids, positions)
            return self.model.compute_logits(h)

    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        return self.sampler(logits, temperatures)
