### Important : Use stable branch

Model compression - matrix multiply with compressed matrix using state-of-the-art ROBE-Z compression.

Speed:
1. This implementation is quite slow and a better implementation is being created for practical usage
2. Update(Feb 22, 22) : Current implementation is 3x slower than complete matrix multiplication for large MM
!! Faster implementation coming soon with measurements of forward and backward passes !!

Notes:
1. Use weight_size > 256 ( SMLS x SMLS) when using TILED=True


Sample Usage:

```
import torch
from RzLinear import RzLinear 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

weight_size = 1000
input_dim = 1000
output_dim = 1000
chunk_size = 2

#hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim), size=((weight_size,))).astype(np.float32)))
hashed_weight = nn.Parameter(torch.from_numpy(np.arange(weight_size).astype(np.float32)))
rzlinear = RzLinear(input_dim, output_dim, chunk_size, hashed_weight).to("cuda:0");

input_v = torch.eye(input_dim).to("cuda:0")
output_v = rzlinear(input_v)

```
