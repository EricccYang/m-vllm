
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator



class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.impl = LinearBackend()

    def forward(self, x: torch.Tensor):
        return self.impl.forward(x,self.weight,self.bias)



class LinearBase(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.ones(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError



class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias,0)


    def weight_loader(self,  param : nn.Parameter, weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim) #直接拿申请好的空间的长度
        start_idx = shard_size * self.tp_rank
        weight = weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.data.copy_(weight)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)





class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias,0)

    def weight_loader(self,  param : nn.Parameter, weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = shard_size * self.tp_rank





class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int,  head_dim: int,  num_heads: int, num_kv_heads: int,  bias: bool = False):
        self.tp_size = dist.get_world_size()
        self.num_heads = num_heads//self.tp_size
        self.num_kv_heads = num_kv_heads//self.tp_size  
        self.head_dim = head_dim
        output_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        super().__init__( input_size ,output_size, bias, 0)
        

    def weight_loader(self,  param : nn.Parameter, weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_dim
            shard_offset  = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_dim
            shard_offset = self.num_heads* self.head_dim
        elif loaded_shard_id == "v":
            shard_size = self.num_kv_heads * self.head_dim
            shard_offset = self.num_heads* self.head_dim + self.num_kv_heads* self.head_dim
        else:
            raise ValueError(f"Invalid loaded_shard_id: {loaded_shard_id}")
        param_data = param_data.narrow(self.tp_dim,shard_offset,shard_size)
        weight = weight.chunk(self.tp_size,self.tp_dim)[self.tp_rank]
        param_data.data.copy_(weight)

        
        



class RowParallelLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self,  param : nn.Parameter, weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = shard_size * self.tp_rank
        weight = weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.data.copy_(weight)

    def forward(self, x: torch.Tensor):
        y = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y