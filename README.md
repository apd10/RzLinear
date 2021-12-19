Model compression - matrix multiply with compressed matrix using state-of-the-art ROBE-Z compression.

Sample Usage:

```
import torch
from RzLinear import RzLinear 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

weight_size = 100
input_dim = 1000
output_dim = 1000
chunk_size = 2

#hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim), size=((weight_size,))).astype(np.float32)))
hashed_weight = nn.Parameter(torch.from_numpy(np.arange(weight_size).astype(np.float32)))
rzlinear = RzLinear(input_dim, output_dim, chunk_size, hashed_weight).to("cuda:0");

input_v = torch.eye(input_dim).to("cuda:0")
output_v = rzlinear(input_v)

```
