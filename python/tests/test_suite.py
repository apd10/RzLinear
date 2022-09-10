import pytest
import torch
import torch.nn as nn
from math import sqrt
import numpy as np
from rz_linear import RzLinear 
from rz_linear import RzLinearFunction 
import pdb
from helpers import *
import matplotlib.pyplot as plt


def test_rz_full():
  output1, output2, wt_grad_rz, wt_grad_ac, in_grad_rz, in_grad_ac = fwdbkd(ishnet=False, init_factor=1.0)
  torch.testing.assert_close(output1, output2, atol=0, rtol=0)
  torch.testing.assert_close(wt_grad_rz, wt_grad_ac, atol=0, rtol=0)
  torch.testing.assert_close(in_grad_rz, in_grad_ac, atol=0, rtol=0)

def test_hnet_full():
  output1, output2, wt_grad_rz, wt_grad_ac, in_grad_rz, in_grad_ac = fwdbkd(ishnet=True, init_factor=1.0)
  torch.testing.assert_close(output1, output2, atol=0, rtol=0)
  torch.testing.assert_close(wt_grad_rz, wt_grad_ac, atol=0, rtol=0)
  torch.testing.assert_close(in_grad_rz, in_grad_ac, atol=0, rtol=0)

def test_rz_full_1():
  init_factor = 2.0
  output1, output2, wt_grad_rz, wt_grad_ac, in_grad_rz, in_grad_ac = fwdbkd(ishnet=False, init_factor=init_factor)
  torch.testing.assert_close(output1, output2, atol=0, rtol=0)
  torch.testing.assert_close(wt_grad_rz, wt_grad_ac, atol=0, rtol=0)
  torch.testing.assert_close(in_grad_rz, in_grad_ac, atol=0, rtol=0)

def test_rz_full_2():
  init_factor = 2.0
  output1, output2, wt_grad_rz, wt_grad_ac, in_grad_rz, in_grad_ac = fwdbkd(ishnet=True, init_factor=init_factor)
  torch.testing.assert_close(output1, output2, atol=0, rtol=0)
  torch.testing.assert_close(wt_grad_rz, wt_grad_ac, atol=0, rtol=0)
  torch.testing.assert_close(in_grad_rz, in_grad_ac, atol=0, rtol=0)

#def test_hnet_full_1():
#  output1, output2, wt_grad_rz, wt_grad_ac, in_grad_rz, in_grad_ac = bwd(ishnet=True, init_factor=2.0)
#  torch.testing.assert_close(output1, output2, atol=0, rtol=0)
#  torch.testing.assert_close(wt_grad_rz, wt_grad_ac, atol=0, rtol=0)
#  torch.testing.assert_close(in_grad_rz, in_grad_ac, atol=0, rtol=0)

# uncomment and plot
#def test_rz_idx():
#  SIZE=1024
#  ids, size = idx(ishnet=False, init_factor=1.0, batch=SIZE, input_dim=SIZE, output_dim=SIZE)
#  ids = np.sort(np.array(ids.view(-1).cpu())) / size
#  plt.hist(ids, bins = int(size/2))
#  plt.show()
#
#def test_hnet_idx():
#  ids, size = idx(ishnet=True, init_factor=1.0, batch=1024, input_dim=1024, output_dim=1024)
#  pdb.set_trace()
#  ids = np.sort(np.array(ids.view(-1).cpu())) / size
#  plt.hist(ids, bins = int(size/2))
#  plt.show()
#
#test_rz_idx()
#test_hnet_idx()

