
import torch
import torch.nn as nn
from math import sqrt
import numpy as np
from rz_linear import RzLinear 
from rz_linear import RzLinearFunction 
import pdb
SEED = 4219
def idx(ishnet, init_factor, batch=128, input_dim=128, output_dim=128):
  MAX=100
  M = batch 
  K = input_dim 
  N = output_dim
  weight_size = 10000
  weight = nn.Parameter(torch.from_numpy(np.arange(weight_size).astype(np.float32)));
  rz = RzLinear(input_dim, output_dim, compress_ratio=0.01, hashed_weight=weight, bias=False, chunk_size=32, seed=SEED, is_hnet=ishnet, init_factor=init_factor).to("cuda:0")
  inn = torch.eye(input_dim).cuda(0)
  IDX = rz(inn).long().cuda(0)
  return IDX, weight_size

def fwdbkd(ishnet, init_factor, batch=128, input_dim=128, output_dim=128):
  MAX=100
  M = batch 
  K = input_dim 
  N = output_dim 
  weight_size = 10000
  weight = nn.Parameter(torch.from_numpy(np.arange(weight_size).astype(np.float32)));
  rz = RzLinear(input_dim, output_dim, compress_ratio=0.01, hashed_weight=weight, bias=False, chunk_size=32, seed=SEED, is_hnet=ishnet, init_factor=init_factor).to("cuda:0")
  inn = torch.eye(input_dim).cuda(0)
  IDX = (rz(inn) / init_factor).long().cuda(0)
  #weight = torch.from_numpy(np.random.uniform(-1/sqrt(100), 1/sqrt(100), size=(weight_size,)).astype(np.float32)).cuda(0)
  weight = nn.Parameter(torch.randint(low=-MAX, high=MAX, size=(weight_size,), device="cuda:0").float());
  rz._hashed_weight.data = weight

  weight2 = torch.nn.Parameter(torch.clone(weight.data), requires_grad=True)

  input1 = torch.nn.Parameter(torch.randint(low=-MAX, high=MAX, size=(M, input_dim), device="cuda:0").float(), requires_grad=True)
  input2 = torch.nn.Parameter(torch.clone(input1.data), requires_grad=True).cuda(0)
  input1.retain_grad()
  input2.retain_grad()

  output1 = rz(input1)
  output2 = init_factor * torch.mm(input2, weight2[IDX])

  loss1 = torch.sum(output1)
  loss2 = torch.sum(output2)

  loss1.backward()
  wt_grad_rz = torch.clone(rz._hashed_weight.grad)
  in_grad_rz = torch.clone(input1.grad)

  loss2.backward()
  wt_grad_ac = torch.clone(weight2.grad)
  in_grad_ac = torch.clone(input2.grad)
  return output1, output2, wt_grad_rz, wt_grad_ac, in_grad_rz, in_grad_ac
