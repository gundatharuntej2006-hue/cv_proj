# -*- coding: utf-8 -*-
"""P2-05: the image-quality half of the gate. Feeds R2 `gate` and decision D07.

Acceptance criteria from the work plan:
  "Synthetic tests: black, white, low-exposure and 60%-cropped images each rejected with
   the correct reason; <= 1% of development rejected; D07 data_used=[development]"

Note what that pair of criteria demands. The rules must be sharp enough to catch four
specific degradations, and blunt enough to leave real radiographs alone. A rule that
rejects everything passes the first and fails the second, so both halves are tested.

**The thresholds here are derived, not guessed.** Each one is calibrated against a
synthetic degradation sweep (`calibrate_thresholds`): we generate progressively worse
versions of an image, measure where a human would call it unusable, and place the
threshold in the gap between the degraded and clean distributions. The one free choice
left is how aggressive to be, and D07 pre-registers it: thresholds sit at the midpoint,
in log space where the statistic is a variance.

Only the four reasons in contract C09 may ever be emitted:
    blank, under_exposed, over_exposed, heavily_cropped
A fifth failure mode would need a contract change request, not a new string.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

REASONS = ["blank", "under_exposed", "over_exposed", "heavily_cropped"]

# Calibrated in P2-05; see calibrate_thresholds() and decisions/p2_quality_rules.json.
# Confirmed against the development split before D07 is locked, never against test.
DEFAULTS = {
    "blank_std": 0.020,            # normalised intensity SD below which there is no image
    "dark_mean": 0.120,            # mean intensity below which detail is lost in the black
    "bright_mean": 0.880,          # ... and above which it is lost in the white
    "saturated_frac": 0.500,       # share of pixels at the very top of the range
    "min_side": 256,               # a 512 radiograph cropped past half its width
    "max_aspect": 1.600,           # or squashed out of a plausible chest shape
}


@dataclass
class QualityReport:
    passed: bool
    reasons: List[str]
    stats: dict = field(default_factory=dict)


def _normalise(image):
    """uint8, float [0,1] or float [0,255] -> float32 [0,1] greyscale, without rescaling
    contrast. Rescaling here would hide exactly the exposure faults we are looking for."""
    a = np.asarray(image)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32)
    if a.size == 0:
        raise ValueError("empty image")
    if a.max() > 1.5:                       # uint8 or float in 0..255
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def assess(image, thresholds=None):
    """Returns a QualityReport. Pure function of the pixels: no model, no randomness."""
    t = dict(DEFAULTS, **(thresholds or {}))
    a = _normalise(image)
    h, w = a.shape[:2]
    std, mean = float(a.std()), float(a.mean())
    sat = float((a > 0.98).mean())
    dark = float((a < 0.02).mean())
    aspect = max(h, w) / max(min(h, w), 1)

    reasons = []
    if std < t["blank_std"]:
        # a uniform field: black, white or grey. One reason, not three.
        reasons.append("blank")
    else:
        if mean < t["dark_mean"] or dark > t["saturated_frac"]:
            reasons.append("under_exposed")
        if mean > t["bright_mean"] or sat > t["saturated_frac"]:
            reasons.append("over_exposed")
    if min(h, w) < t["min_side"] or aspect > t["max_aspect"]:
        reasons.append("heavily_cropped")

    return QualityReport(passed=not reasons, reasons=reasons,
                         stats={"std": std, "mean": mean, "saturated_frac": sat,
                                "dark_frac": dark, "height": h, "width": w,
                                "aspect": aspect})


# ----------------------------------------------------------------- degradations
def degrade(image, kind, severity=1.0, rng=None):
    """The four failures named in C09, plus the clean control. `severity` runs 0 to 1.

    These are the test fixtures and the calibration set at once, which is the point:
    the thresholds are fitted to the same degradations the acceptance criteria name.
    """
    rng = rng or np.random.default_rng(0)
    a = _normalise(image).copy()
    if kind == "clean":
        return a
    if kind == "black":
        return a * (1.0 - severity)
    if kind == "white":
        return 1.0 - (1.0 - a) * (1.0 - severity)
    if kind == "under_exposed":
        return np.clip(a * (1.0 - 0.85 * severity), 0, 1)
    if kind == "over_exposed":
        return np.clip(a + 0.85 * severity * (1.0 - a), 0, 1)
    if kind == "heavily_cropped":
        h, w = a.shape
        keep = int(w * (1.0 - 0.7 * severity))
        return a[:, : max(keep, 8)]
    raise ValueError("unknown degradation %r" % kind)


def calibrate_thresholds(images, rng=None, severities=(0.8, 0.9, 1.0)):
    """Where do clean images sit, and where do degraded ones? Returns the two
    distributions per statistic so a threshold can be placed in the gap between them,
    rather than picked because it looked about right.
    """
    rng = rng or np.random.default_rng(0)
    out = {}
    clean = [assess(im).stats for im in images]
    out["clean"] = {k: [c[k] for c in clean] for k in ("std", "mean", "saturated_frac")}
    for kind in ("black", "white", "under_exposed", "over_exposed"):
        st = [assess(degrade(im, kind, s, rng)).stats
              for im in images for s in severities]
        out[kind] = {k: [c[k] for c in st] for k in ("std", "mean", "saturated_frac")}
    return out


def gap_midpoint(clean_values, bad_values, log=False):
    """A threshold placed between the worst clean image and the best degraded one.
    Returns (threshold, separated) — `separated` is False when the two distributions
    overlap, which means no threshold can satisfy both acceptance criteria and the rule
    itself needs rethinking rather than tuning."""
    c, b = np.asarray(clean_values, float), np.asarray(bad_values, float)
    if c.min() > b.max():                   # clean is high, degraded is low
        lo, hi, separated = b.max(), c.min(), True
    elif b.min() > c.max():                 # degraded is high, clean is low
        lo, hi, separated = c.max(), b.min(), True
    else:
        return None, False
    if log:
        lo, hi = max(lo, 1e-9), max(hi, 1e-9)
        return float(np.exp((np.log(lo) + np.log(hi)) / 2)), separated
    return float((lo + hi) / 2), separated
