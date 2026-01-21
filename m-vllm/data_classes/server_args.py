from dataclasses import dataclass

@dataclass
class ServerArgs:
    tensor_parallel_size: int
    model_path: str
    