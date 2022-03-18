Model compression - matrix multiply with compressed matrix using state-of-the-art ROBE-Z compression.

### Notes

Use M, N, K % 32 == 0

### Install

```
pip install .
```

### Sample Usage

```
import torch
from rz_linear import RzLinear

N = 128
input_dim = 1024
output_dim = 1024

rz = RzLinear(input_dim, output_dim).to("cuda:0")
input = torch.rand((N, input_dim), device="cuda:0")
output = rz(input)
```
