"""Seed every RNG through tbcore when available (P4, tbcore.seed.set_all), else a local fallback.

Returns which one was used, so the caller can record it next to the seed (rule: every training
run records its seed).
"""
from __future__ import annotations

import random


def set_seed(seed: int) -> str:
    try:
        from tbcore.seed import set_all
    except ImportError:
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return "fallback (tbcore.seed not available)"
    set_all(seed)
    return "tbcore.seed.set_all"
