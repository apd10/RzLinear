
Installation :
Our ROAST-MM implementation has issues with latest trition. We are working to fix this. However, the code can be used with previous version of trition which can be locally compiled. 

0. Install latest pytorch from pytorch.org

1. Local Installation of triton
```
git clone https://github.com/apd10/triton
cd triton/python
pip install -e . 
```

2. Install RzLinear (ROAST-MM)
```
git clone https://github.com/apd10/RzLinear 
git checkout stable # be on stable branch
cd RzLinear/python/
pip install -e .
```

3. Quick test (outputs should be less than 1e-5)
```
python3 quick_tests.py
```




Sample Usage:

```
import torch
from rz_linear import RzLinear 
import torch.nn as nn
import numpy as np


M = 32
K = input_dim = 128
N = output_dim = 128
weight_size = 10000
scale=2.0

weight = torch.from_numpy(np.random.uniform(-1/np.sqrt(100000), 1/np.sqrt(100000), size=(weight_size,)).astype(np.float32)).cuda(0)   # global weight array 
rz = RzLinear(input_dim, output_dim, compress_ratio=0.01, hashed_weight=weight, bias=False, chunk_size=32, seed=1367, init_factor=scale).to("cuda:0") # ROAST-MM module with scale
input_v = torch.eye(input_dim).to("cuda:0")
output_v = rz(input_v)

```
