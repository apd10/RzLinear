import re
from math import floor
from typing import Union


def cast_bytes_to_memory_string(num_bytes: float) -> str:

    suffix = "B"
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_bytes) < 1024.0:
            return "%3.1f%s%s" % (num_bytes, unit, suffix)
        num_bytes /= 1024.0
    return "%.1f%s%s" % (num_bytes, "Y", suffix)
