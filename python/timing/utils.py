from functools import wraps
import torch


knobs = {}


def set_device(dev: str):
    knobs['device'] = dev


def get_device():
    return knobs['device']


def set_verbose(flag: bool):
    knobs['verbose'] = flag


def get_model_bytes(model) -> str:
    # Assuming
    num_bytes = sum([p.storage().element_size() * p.numel()
                    for p in model.parameters()])
    suffix = "B"
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_bytes) < 1024.0:
            return "%3.1f%s%s" % (num_bytes, unit, suffix)
        num_bytes /= 1024.0
    return "%.1f%s%s" % (num_bytes, "Y", suffix)


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


def vprint(msg):
    if knobs['verbose']:
        print(msg)
