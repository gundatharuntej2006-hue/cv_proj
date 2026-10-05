# General helper utilities: seeds, timer, and hashing.

import time
import random
import hashlib
from typing import Callable, Any, Tuple
import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def timed_execution(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return res, elapsed_ms
