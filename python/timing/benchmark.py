""" Benchmark on compressed embedding table speed """

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse

from typing import List
from simple_model import SimpleModel
from autotuning import autotune, load_configs
from utils import *

from tqdm import tqdm as tq
from tabulate import tabulate

from rz_linear.RzLinearFunction import controls


DEFAULT_MEM_SIZES = [1024*1024, 1024*1024*8, 1024 *
                     1024*16, 1024*1024*32, 1024*1024*64, 1024*1024*128]
MAX_ITERS = 10
WARMUP_ITERS = 2


def benchmark(shapes: List[List[int]], batch_sizes: List[int], mem_sizes: List[int], optimizers: List[str], mode:str) -> pd.DataFrame:
    """ Benchmark compressed embedding table speed """
    report = pd.DataFrame()
    model_names = ["Full", "HNet", "ROBE-Sketch"]
    hnet = [None, True, False]
    loss_fct = nn.MSELoss().to(device=get_device())
    if mode == "forward":
        optimizers = ["none"]

    for optimizer_name in optimizers:
        for bs in tq(batch_sizes):
            for shape in shapes:
                for i in range(len(model_names)):
                    model_name = model_names[i]
                    is_hnet = hnet[i]
                    is_robez = (model_name != "Full")
                    for j, mem in enumerate(mem_sizes):
                        if model_name == "Full":
                            mem = shape[0] * shape[1]
                            if j > 0:
                                continue

                        model = SimpleModel(
                            shape[0], shape[1], mem, is_robez, is_hnet).to(get_device())
                        if optimizer_name == "sgd":
                            optimizer = torch.optim.SGD(
                                model.parameters(), lr=0.001)
                        elif optimizer_name == "adagrad":
                            optimizer = torch.optim.Adagrad(
                                model.parameters(), lr=0.001)
                        elif optimizer_name == "adam":
                            optimizer = torch.optim.Adam(
                                model.parameters(), lr=0.001)
                        elif optimizer_name !=  "none":
                            raise NotImplementedError

                        if mode == "forward":
                            model.eval()
                        else:
                            model.train()

                        def train(iters, x, y, forward_times=None, backward_times=None, opt_times=None):
                            for _ in range(iters):
                                # forward
                                t, y_pred = timing(model)(x)
                                if forward_times is not None:
                                    forward_times.append(t)
                                # loss
                                loss_value = loss_fct(y, y_pred)
                                # grad
                                t, _ = timing(loss_value.backward)()
                                if backward_times is not None:
                                    backward_times.append(t)
                                # opt
                                t, _ = timing(optimizer.step)()
                                if opt_times is not None:
                                    opt_times.append(t)
                                optimizer.zero_grad()

                        def eval(iters, x, y, forward_times=None):
                            for _ in range(iters):
                                # forward
                                t, y_pred = timing(model)(x)
                                if forward_times is not None:
                                    forward_times.append(t)
                                # loss
                                loss_value = loss_fct(y, y_pred)

                        x = torch.rand((bs, shape[0]), device=get_device())
                        y = torch.rand((bs, 1), device=get_device())

                        forward_pass = []
                        backward_pass = []
                        opt_computation = []

                        if mode == "forward":
                            eval(WARMUP_ITERS, x, y)
                            eval(MAX_ITERS, x, y, forward_pass)
                            forward_pass = np.median(forward_pass[:])
                            backward_pass = 0
                            opt_computation = 0
                        else:
                            train(WARMUP_ITERS, x, y)
                            train(MAX_ITERS, x, y, forward_pass,
                                  backward_pass, opt_computation)

                            forward_pass = np.median(forward_pass[:])
                            backward_pass = np.median(backward_pass[:])
                            opt_computation = np.median(opt_computation[:])

                        infos = pd.DataFrame(
                            {
                                "mode": mode,
                                "Model": model_name,
                                "bs": [bs],
                                "I": [shape[0]],
                                "O": [shape[1]],
                                "mem_params": [mem],
                                "forward(ms)": [forward_pass],
                                "backward(ms)": [backward_pass],
                                "optcomp(ms)": [opt_computation],
                                "total(ms)": [forward_pass + backward_pass + opt_computation],
                                "msize": [get_model_bytes(model)],
                                "optim": [optimizer_name],
                            }
                        )
                        vprint(infos)
                        report = pd.concat([report, infos])

    return report


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--autotune', action='store_true', default=False)
parser.add_argument('-tf32', '--tf32', action='store_true', default=False)
parser.add_argument('-op', '--optimizer',
                    choices=['all', 'sgd', 'adagrad', 'adam'], default='all')
parser.add_argument('-s', '--save', type=str,
                    help='Save autotuning configurations', required=False, default='')
parser.add_argument('-l', '--load', type=str,
                    help='Load autotuning configurations', required=False, default='')
parser.add_argument('-o', '--output', type=str,
                    help='Output benchmark file', default='benchmark.csv')
parser.add_argument('-d', '--dims', type=str, default='10240',
                    help='matrix feature dims, 1024x1024,512x512 separated by , ')
parser.add_argument('-b', '--batch-sizes', type=str,
                    default='512', help='batch sizes, separated by')
parser.add_argument('-i', '--iterations', type=int, required=False)
parser.add_argument('-c', '--cuda', type=str, default='cuda:0')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-m', '--mode', type=str, default="forward+backward",
                             help= 'string in [forward, backward, forward+backward],\
                                what functions to run for autotuning')
# TODO(Keren): make mem_sizes configurable
args = parser.parse_args()


def str_to_int_list(s, d=','):
    l = s.split(d)
    res = map(int, l)
    return list(res)


def dims_to_shapes(dims):
    return [[int(dim.split("x")[0]), int(dim.split("x")[1])] for dim in dims.split(',')]


set_verbose(args.verbose)
set_device(args.cuda)

controls['triton_allow_autotune'] = False

shapes = dims_to_shapes(args.dims)
batch_sizes = str_to_int_list(args.batch_sizes)

if args.tf32:
    controls['triton_allow_tf32'] = True
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    controls['triton_allow_tf32'] = False
    torch.backends.cuda.matmul.allow_tf32 = False

if args.load != '':
    load_configs(file_name=args.load)
elif args.autotune:
    autotune(batch_sizes=batch_sizes, shapes=shapes, mem_sizes=DEFAULT_MEM_SIZES,
             file_name=args.save, allow_tf32=controls['triton_allow_tf32'], mode=args.mode)

if args.optimizer == 'all':
    optimizers = ['sgd', 'adagrad', 'adam']
else:
    optimizers = [args.optimizer]

if args.iterations is not None:
    MAX_ITERS = args.iterations
    WARMUP_ITERS = 0

report = benchmark(shapes=shapes, batch_sizes=batch_sizes,
                   mem_sizes=DEFAULT_MEM_SIZES, optimizers=optimizers, mode=args.mode)
report.to_csv(args.output, index=False)

vprint(tabulate(report, headers='keys', tablefmt='github'))
