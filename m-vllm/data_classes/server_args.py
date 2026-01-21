from dataclasses import dataclass

@dataclass
class ServerArgs:
    model_path: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    