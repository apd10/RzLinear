from contextlib import contextmanager
from functools import wraps
import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        te = time.perf_counter()
        return te - ts, result
    return wrap
