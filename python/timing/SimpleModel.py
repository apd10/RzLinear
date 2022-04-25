import torch
import torch.nn as nn
from rz_linear import  RzLinear
import numpy as np


class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, mem: int, robez: bool, hnet: bool):
        super(SimpleModel, self).__init__()
        if robez:
            self.weight = nn.Parameter(torch.from_numpy(np.arange(mem).astype(np.float32)));
            self.mm = RzLinear(input_dim, output_dim, hashed_weight=self.weight, bias=False, chunk_size=32, seed=1367, is_hnet=hnet)
        else:
            self.mm = torch.nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = self.mm(x)
        x = torch.norm(x, dim=1).view(-1,1)
        return x
