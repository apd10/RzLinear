""" Benchmark on compressed embedding table speed """

import torch
import torch.nn as nn
from typing import List
from SimpleModel import SimpleModel
from converter import cast_bytes_to_memory_string
from decorators import timing

from tqdm import tqdm as tq
import pandas as pd
import numpy as np
import pdb
from tabulate import tabulate

from rz_linear.RzLinearFunction import controls

MAX_ITERS = 22
WARMUP_ITERS = 5

def count_parameters(model):
    num = np.sum([p.numel() for p in model.parameters()])
    return num



def benchmark(shapes : List[List[int]], batchsizes: List[int], mem_size: List[int]) -> pd.DataFrame:
    """ Benchmark compressed embedding table speed """

    report = pd.DataFrame()
    model_names = ["Full", "HNet", "ROBE-Sketch"]
    hnet = [None,True,False]

    optimizers = [
        'sgd',
        'adagrad',
        'adam'
    ]

    for optimizer_name in optimizers:
        for bs in tq(batch_sizes):
            for shape in shapes:
                for i in range(len(model_names)):
                    model_name = model_names[i]
                    is_hnet = hnet[i]
                    is_robez = (model_name != "Full")
                    for j, mem in enumerate(mem_size):
                        if model_name == "Full":
                            mem = shape[0] * shape[1]
                            if j > 0:
                                continue
                        model = SimpleModel(shape[0], shape[1], mem, is_robez, is_hnet).cuda(0)
                        if optimizer_name == "sgd":
                            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                        elif optimizer_name == "adagrad":
                            optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
                        elif optimizer_name == "adam":
                            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        else:
                            raise NotImplementedError
    
                        loss_fct = nn.MSELoss().cuda(0)
    
                        forward_pass = []
                        backward_pass = []
                        opt_computation = []
                        model.train()
                        
                        for i in range(22):
                            x = torch.from_numpy(np.random.rand(bs, shape[0])).float().cuda(0)
                            y = torch.from_numpy(np.random.uniform(size=(bs, 1))).float().cuda(0)
                            
                            # forward
                            t, y_pred = timing(model)(x)
                            if i >= WARMUP_ITERS:
                                forward_pass.append(t)
                            # loss
                            loss_value = loss_fct(y, y_pred)
                            # grad
                            t,_ = timing(loss_value.backward)()
                            if i >= WARMUP_ITERS:
                                backward_pass.append(t)
                            # opt
                            t, _ = timing(optimizer.step)()
                            if i >= WARMUP_ITERS:
                                opt_computation.append(t)
                            optimizer.zero_grad()
    
                        forward_pass = np.median(forward_pass[2:])
                        backward_pass = np.median(backward_pass[2:])
                        opt_computation = np.median(opt_computation[2:])

    
                        infos = pd.DataFrame(
                            {
                                "Model": model_name,
                                "bs": [bs],
                                "I": [shape[0]],
                                "O" : [shape[1]],
                                "mem_params": [mem],
                                "forward(ms)": [forward_pass],
                                "backward(ms)": [backward_pass],
                                "optcomp(ms)": [opt_computation],
                                "total(ms)": [forward_pass + backward_pass + opt_computation],
                                "msize": [cast_bytes_to_memory_string(4 * count_parameters(model))],
                                "optim": [optimizer_name],
                            }
                        )
                        print(infos)
                        report = pd.concat([report, infos])
                        report.to_csv("temp.csv", index=False)

    return report


# pylint: disable=redefined-outer-name
if __name__ == "__main__":
    controls['triton_allow_autotune'] = True
    #benchmark(shapes : List[List[int]], batchsizes: List[int], mem_size: List[int]) -> pd.DataFrame:
    #shapes = [[1024, 1024], [10240,10240]]
    shapes = [[10240, 10240]]
    #batch_sizes = [64, 512, 4096, 32768]
    #batch_sizes = [64, 512, 1024, 4096, 10240]
    #batch_sizes = [64, 512, 4096]
    batch_sizes = [512]

    #mem_size = [1, 8, 64, 512] + [32768 * 8 ** i for i in range(5)] + [nb_embeddings*embeding_dim]
    #mem_size = [64, 512] + [32768 * 8 ** i for i in range(5)]
    mem_size=[1024*1024, 1024*1024*8, 1024*1024*16, 1024*1024*32, 1024*1024*64, 1024*1024*128]
    #mem_size=[1024*1024, 1024*1024*8]

    my_report = benchmark(shapes, batch_sizes, mem_size)
    my_report.to_csv("benchmark.csv", index=False)
    print(tabulate(my_report, headers='keys', tablefmt='github'))
