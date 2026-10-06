"""Deterministic seed utility for cross-library reproducibility."""

import os
import random
import numpy as np


def set_all(seed: int = 42) -> int:
    """Sets deterministic seed across Python random, NumPy, and PyTorch (if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
