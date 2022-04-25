from contextlib import contextmanager
from functools import wraps
import time
import torch


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        torch.cuda.synchronize()
        ts = time.perf_counter()
        result = f(*args, **kw)
        torch.cuda.synchronize()
        te = time.perf_counter()
        return te - ts, result
    return wrap
