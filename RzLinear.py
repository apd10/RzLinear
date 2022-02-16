from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math
import rz_linear
import pdb

class RzLinearFunction(torch.autograd.Function):
    @staticmethod

    def forward(ctx, hashed_weights, input_v, random_numbers, input_dim, output_dim, chunk_size, tiled):
        '''
            read a chunk_size by performing lsh according to the lsh_mode,
            join chunks to create an embedding of size embedding_dim for each of the
            inputs
        '''
        output = rz_linear.forward(hashed_weights, input_v, random_numbers,  input_dim, output_dim, chunk_size, tiled)
        ctx.save_for_backward(hashed_weights, input_v, random_numbers)
        ctx.input_dim =  input_dim
        ctx.output_dim = output_dim
        ctx.chunk_size =  chunk_size
        ctx.tiled = tiled
        return output

    @staticmethod
    def backward(ctx, grad):
        hashed_weights, input_v, random_numbers = ctx.saved_variables
        input_dim = ctx.input_dim
        output_dim = ctx.output_dim
        chunk_size = ctx.chunk_size
        in_grad, wt_grad = rz_linear.backward(grad, hashed_weights, input_v, random_numbers, input_dim, output_dim, chunk_size, ctx.tiled)
        return wt_grad, in_grad, None, None, None, None, None
    @staticmethod
    def forwardproxy(hashed_weights, input_v, random_numbers, input_dim, output_dim, chunk_size, tiled):
        output = rz_linear.forward(hashed_weights, input_v, random_numbers,  input_dim, output_dim, chunk_size, tiled)
        return output
  
    @staticmethod
    def backwardproxy(grad, hashed_weights, input_v, random_numbers, input_dim, output_dim, chunk_size, tiled):
        in_grad, wt_grad = rz_linear.backward(grad, hashed_weights, input_v, random_numbers, input_dim, output_dim, chunk_size, tiled)
        return in_grad, wt_grad

    @staticmethod
    def get_idx(random_numbers, input_dim, output_dim, chunk_size, weight_size, tiled):
        return rz_linear.get_idx(random_numbers, input_dim, output_dim, chunk_size, weight_size, tiled)

class RzLinear(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        chunk_size: int,
        hashed_weight: torch.Tensor,
        tiled = True,
        seed = 1024)->None:
        super(RzLinear, self).__init__()


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.chunk_size = chunk_size
        self.weight = hashed_weight
        self.memory_size = hashed_weight.size(0)
        self.tiled = tiled

        r = np.random.RandomState(seed)
        # first number is the prime, rest are random integers
        x = r.randint(0, 2038074743, (50,))
        x = x + 1*(x%2==0);
        random_numbers = np.concatenate([np.array([2038074743]), x]) # set of 50 random numbers to use
        self.random_numbers = Parameter(torch.from_numpy(random_numbers.astype(np.int64)), requires_grad=False)

        # bias term
        self.bias = Parameter(torch.zeros(self.output_dim, ))

        print("RandomNumbers: ", self.random_numbers[:5])
        print("RzLinear: d1xd2: {}x{} chunk_size: {} weight_size: {}  tiled: {}".format(self.input_dim, self.output_dim, self.chunk_size, self.weight.shape[0], self.tiled))
        

    def forward(self, input_v) -> torch.Tensor:
        dim_gt_2 = input_v.dim() > 2 
        x = input_v
        if (dim_gt_2):
            shape = input_v.shape
            x = input_v.view(-1, shape[-1])
        output_v =  RzLinearFunction.apply(self.weight, x, self.random_numbers, self.input_dim, self.output_dim, self.chunk_size, self.tiled)
        output_v = output_v + self.bias
        if (dim_gt_2):
            output_v = output_v.view(*shape[:-1], output_v.shape[-1])
        return output_v
