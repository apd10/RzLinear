import torch
from rz_linear import RzLinear 
from rz_linear import RzLinearFunction 
import torch.nn as nn
import numpy as np
#import matplotlib.pyplot as plt
import rz_linear
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import pdb


M = 32
K = input_dim = 128
N = output_dim = 128
weight_size = 10000
scale=2.0

weight = nn.Parameter(torch.from_numpy(np.arange(weight_size).astype(np.float32) / scale));
rz = RzLinear(input_dim, output_dim, compress_ratio=0.01, hashed_weight=weight, bias=False, chunk_size=32, seed=1367, init_factor=scale).to("cuda:0")
inn = torch.eye(input_dim).cuda(0)
IDX = rz(inn).long().cuda(0)

weight = torch.from_numpy(np.random.uniform(-1/np.sqrt(100000), 1/np.sqrt(100000), size=(weight_size,)).astype(np.float32)).cuda(0)
rz._hashed_weight.data = weight
W = nn.Parameter(weight[IDX], requires_grad=True).cuda(0)

print(rz)
input1 = torch.nn.Parameter(torch.rand((M, input_dim), device="cuda:0"), requires_grad=True)
input2 = torch.nn.Parameter(torch.clone(input1.data), requires_grad=True).cuda(0)
input1.retain_grad()
input2.retain_grad()

output1 = rz(input1)
output2 = scale * torch.mm(input2, W)

loss1 = torch.sum(output1)
loss2 = torch.sum(output2)

print("MM - forward")
print(torch.max(torch.abs(output1 - output2)), torch.mean(torch.abs(output1)))

loss1.backward()
wt_grad_rz = torch.clone(rz._hashed_weight.grad)
in_grad_rz = torch.clone(input1.grad)

loss2.backward()
wt_grad_ac = torch.clone(W.grad)
in_grad_ac = torch.clone(input2.grad)
wt_grad_actual_p = torch.zeros_like(rz._hashed_weight).cuda(0)
wt_grad_actual_p.scatter_add_(0, IDX.reshape(-1), wt_grad_ac.reshape(-1))


print("MM - backward Input")
print(torch.max(torch.abs(in_grad_ac - in_grad_rz)), torch.mean(torch.abs(in_grad_rz)))


print("MM - backward W")
print(torch.max(torch.abs(wt_grad_actual_p - wt_grad_rz)), torch.mean(torch.abs(wt_grad_rz)))


#plt.hist(np.array(IDX.detach().cpu()).reshape(-1), bins=100)
#plt.show()
