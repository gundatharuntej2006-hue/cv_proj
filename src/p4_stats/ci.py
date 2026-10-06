"""Confidence interval computation: Wilson score interval and stratified bootstrap."""

from typing import Callable, Optional, Tuple, Union
import numpy as np
from scipy import stats


def wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Computes exact Wilson score confidence interval for a binomial proportion k / n.
    Matches statsmodels proportion_confint(k, n, method='wilson').
    
    Returns:
        (ci_low, ci_high) in [0.0, 1.0]
    """
    if n <= 0:
        return 0.0, 1.0

    p_hat = k / n
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    if k <= 0:
        return 0.0, float(min(1.0, center + half_width))
    if k >= n:
        return float(max(0.0, center - half_width)), 1.0

    ci_low = float(max(0.0, center - half_width))
    ci_high = float(min(1.0, center + half_width))

    return ci_low, ci_high


def bootstrap_ci(
    data: np.ndarray | list,
    stat_fn: Callable[[np.ndarray], float],
    B: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    stratified: bool = False,
    y: Optional[np.ndarray | list] = None
) -> Tuple[float, float, float]:
    """
    Computes empirical point estimate and bootstrap percentile confidence interval.
    
    Args:
        data: input dataset (array-like, e.g. scores or indices)
        stat_fn: callable taking resampled data array and returning scalar statistic
        B: number of bootstrap replicates (default 2000)
        confidence: confidence level (default 0.95)
        seed: fixed random seed for reproducibility
        stratified: if True, performs class-stratified sampling using y
        y: binary class labels required if stratified=True
        
    Returns:
        (point_estimate, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    data_arr = np.asarray(data)
    point_est = float(stat_fn(data_arr))

    n = len(data_arr)
    if n == 0:
        return point_est, 0.0, 1.0

    boot_stats = np.empty(B, dtype=np.float64)

    if stratified and y is not None:
        y_arr = np.asarray(y)
        pos_indices = np.where(y_arr == 1)[0]
        neg_indices = np.where(y_arr == 0)[0]
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)

        for b in range(B):
            resamp_pos = rng.choice(pos_indices, size=n_pos, replace=True)
            resamp_neg = rng.choice(neg_indices, size=n_neg, replace=True)
            resamp_idx = np.concatenate([resamp_pos, resamp_neg])
            boot_stats[b] = stat_fn(data_arr[resamp_idx])
    else:
        for b in range(B):
            resamp_idx = rng.choice(n, size=n, replace=True)
            boot_stats[b] = stat_fn(data_arr[resamp_idx])

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_low = float(np.percentile(boot_stats, lower_pct))
    ci_high = float(np.percentile(boot_stats, upper_pct))

    return point_est, ci_low, ci_high
