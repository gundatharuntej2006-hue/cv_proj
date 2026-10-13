# -*- coding: utf-8 -*-
"""P2-11: the kNN out-of-distribution gate. Feeds R2 `gate` (OOD half), C09 and D13.

Methodology: score = mean cosine *distance* from an image's L2-normalised RAD-DINO CLS
embedding to its k=5 nearest neighbours in a bank of one unaugmented embedding per training
image. An image is OOD if its score exceeds a threshold.

Two things this module gets right that the throwaway prototype did not, both learned the
hard way in this project:

**The threshold is set conformally, and the gate abstains when it cannot honour the
guarantee.** Treating the calibration images as exchangeable, the threshold at level alpha
is the `ceil((n+1)(1-alpha))`-th smallest calibration score. This gives a finite-sample
guarantee that P(flag a genuine in-distribution image) <= alpha. It is defined only when
`n >= 1/alpha - 1` (nine images at alpha=0.10, nineteen at alpha=0.05); below that, no
threshold can honour the guarantee and `fit_threshold` returns None rather than a number
it cannot justify. This is the identical bug that was found in Person 4's
`compute_conformal_lower_threshold` during review — a conformal routine must abstain, not
invent, below its minimum sample size.

**The threshold does not transfer between sites; the score does.** Measured on a genuinely
independent site (CheXpert), a threshold fitted on TBX11K rejects 98.7% of ordinary chest
X-rays, while the score still separates true OOD from normal at AUROC 0.98. The remedy is to
re-fit the threshold on a small sample of local normal images — about 25 — at the new site.
`fit_threshold` is therefore the operation run at deployment, not only at training. See
`docs/p2_ood_gate_finding.md`.

The D13 default keeps the methodology's "95% development pass" rule, which is the special
case alpha = 0.05 applied to the development split at the home site.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_K = 5
DEFAULT_ALPHA = 0.05            # 95% in-distribution pass, the methodology's home-site rule


def l2_normalise(emb):
    x = np.asarray(emb, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class OodBank:
    """The training bank and the scoring function. One unaugmented embedding per training
    image, L2-normalised (C07 guarantees normalisation, but we enforce it so a bank built
    from any source behaves identically)."""

    def __init__(self, bank_emb, image_ids=None, k=DEFAULT_K):
        self.bank = l2_normalise(bank_emb)
        self.image_ids = list(image_ids) if image_ids is not None else None
        self.k = int(k)
        if len(self.bank) < self.k:
            raise ValueError("bank has %d vectors but k=%d" % (len(self.bank), self.k))

    def score(self, emb, k=None, block=512, exclude_self=False):
        """Mean cosine distance to the k nearest bank vectors. `exclude_self` drops the
        top match, for scoring the bank against itself without every image finding itself
        at distance zero."""
        k = self.k if k is None else int(k)
        q = l2_normalise(emb)
        out = np.empty(len(q), dtype=np.float32)
        kk = k + 1 if exclude_self else k
        for s in range(0, len(q), block):
            sim = q[s:s + block] @ self.bank.T
            top = np.partition(sim, -kk, axis=1)[:, -kk:]
            top = np.sort(top, axis=1)                     # ascending
            if exclude_self:
                top = top[:, :-1]                          # drop the nearest (itself)
            out[s:s + block] = 1.0 - top[:, -k:].mean(axis=1)
        return out


# --------------------------------------------------------------- conformal threshold
def min_calibration_n(alpha):
    """Smallest n for which a conformal threshold at level alpha exists."""
    return int(math.ceil(1.0 / alpha - 1.0))


def fit_threshold(calib_scores, alpha=DEFAULT_ALPHA):
    """Conformal OOD threshold from in-distribution calibration scores.

    Returns the threshold, or **None** when n < 1/alpha - 1, meaning no threshold can
    guarantee a false-flag rate at or below alpha and the gate must abstain. Callers treat
    None as "cannot gate at this site with this little data", never as "accept everything".
    """
    s = np.sort(np.asarray(calib_scores, dtype=np.float64))
    n = len(s)
    if n == 0 or n < min_calibration_n(alpha):
        return None
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:                                           # guard; equivalent to the n check
        return None
    return float(s[rank - 1])


@dataclass
class ThresholdFit:
    threshold: Optional[float]
    alpha: float
    n_calibration: int
    k: int
    abstained: bool
    reason: str = ""


def fit(bank: OodBank, calib_emb, alpha=DEFAULT_ALPHA, exclude_self=False) -> ThresholdFit:
    """Fit the gate threshold on in-distribution calibration embeddings. This is the
    operation run both at training (home site) and at deployment (a new site), with the
    calibration set drawn from that site."""
    scores = bank.score(calib_emb, exclude_self=exclude_self)
    t = fit_threshold(scores, alpha)
    n = len(scores)
    if t is None:
        return ThresholdFit(None, alpha, n, bank.k, True,
                            "n=%d < minimum %d for alpha=%.2f; gate abstains"
                            % (n, min_calibration_n(alpha), alpha))
    return ThresholdFit(t, alpha, n, bank.k, False)


def is_ood(scores, threshold):
    """A score strictly above the threshold is out-of-distribution. With a None threshold
    the gate has abstained and flags nothing as OOD (the quality half of the gate and the
    downstream decision-support framing remain in force)."""
    if threshold is None:
        return np.zeros(len(np.atleast_1d(scores)), dtype=bool)
    return np.asarray(scores, dtype=np.float32) > threshold


# --------------------------------------------------------------- empirical check
def false_flag_rate(bank: OodBank, test_in_dist_emb, threshold, exclude_self=False):
    """Share of genuine in-distribution images the threshold would flag. Should sit at or
    below alpha on exchangeable data; this is what the conformal guarantee promises."""
    if threshold is None:
        return 0.0
    s = bank.score(test_in_dist_emb, exclude_self=exclude_self)
    return float((s > threshold).mean())
