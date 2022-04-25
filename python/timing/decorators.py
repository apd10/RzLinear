from contextlib import contextmanager
from functools import wraps
import time
import torch


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = f(*args, **kw)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        return t, result
    return wrap
