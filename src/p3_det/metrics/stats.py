"""P3-05: confidence intervals for detection metrics.

* Wilson score interval for simple proportions over images (methodology 12: proportions carry
  95% Wilson CIs).
* Percentile bootstrap over IMAGES (never boxes), stratified by TB / non-TB image, fixed seed
  (methodology 12: AP carries a 95% bootstrap CI; D21: B = 2,000, stratified, fixed seed).

These mirror what P4 will own in p4_stats; once p4_stats exists, P4 may swap them in.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

DEFAULT_B = 2000
DEFAULT_SEED = 0
D21_PATH = Path(__file__).resolve().parents[3] / "decisions" / "p4_analysis_plan.json"


def bootstrap_settings(path: Path = D21_PATH) -> tuple[int, int, str]:
    """(B, seed, source). Reads locked D21 ``value.bootstrap.{B, seed}`` if present, else defaults.

    The key names are an assumption until P4 publishes D21; a missing key falls back with a warning.
    """
    try:
        d = json.loads(path.read_text())
        b = d["value"]["bootstrap"]
        if d.get("locked"):
            return int(b["B"]), int(b["seed"]), "D21"
    except (OSError, ValueError, KeyError, TypeError):
        pass
    warnings.warn(f"D21 not locked/readable: bootstrap B={DEFAULT_B}, seed={DEFAULT_SEED} (defaults)",
                  stacklevel=2)
    return DEFAULT_B, DEFAULT_SEED, "default"


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for successes / n."""
    if n <= 0:
        raise ValueError("Wilson CI needs n > 0")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(n_items: int, statistic: Callable[[np.ndarray], float], *, strata: Sequence | None = None,
                 B: int = DEFAULT_B, seed: int = DEFAULT_SEED, alpha: float = 0.05) -> tuple[float, float, int]:
    """Percentile bootstrap CI of ``statistic(indices)`` resampling items (images) with replacement.

    ``strata`` (one label per item, e.g. "tb"/"non_tb") keeps each stratum's size fixed in every
    replicate. Replicates where the statistic is undefined (NaN) are dropped; their count is
    returned so the caller can report it. Returns (low, high, n_dropped).
    """
    if n_items <= 0:
        raise ValueError("bootstrap needs at least one image")
    rng = np.random.default_rng(seed)
    groups = [np.arange(n_items)] if strata is None else [
        np.flatnonzero(np.asarray(strata) == s) for s in dict.fromkeys(strata)]
    values = np.empty(B)
    for i in range(B):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        values[i] = statistic(idx)
    ok = values[~np.isnan(values)]
    dropped = B - len(ok)
    if not len(ok):
        return math.nan, math.nan, dropped
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), dropped
