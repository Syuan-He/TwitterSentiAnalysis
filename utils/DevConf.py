from dataclasses import dataclass
import torch

@dataclass
class DevConf:
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f'Unsupported device: {self.device}')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available')