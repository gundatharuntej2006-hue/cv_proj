"""Seed every RNG through tbcore when available (P4, tbcore.seed.set_all, via the adapter), else a local fallback.

Returns which one was used, so the caller can record it next to the seed (rule: every training
run records its seed).
"""
from __future__ import annotations

import random

from . import _tbcore_adapter


def set_seed(seed: int) -> str:
    if _tbcore_adapter.seed_all(seed):
        return "tbcore.seed.set_all"
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return "fallback (tbcore.seed not available)"
