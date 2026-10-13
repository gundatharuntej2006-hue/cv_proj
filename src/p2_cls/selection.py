# -*- coding: utf-8 -*-
"""P2-08 / P2-09: preprocessing and head selection on the development split.

P2-08 chooses CLAHE on/off (D08) and the crop margin (D09); P2-09 chooses the head,
binary vs three-class (D11). All three are decided **on development only**, by a rule
fixed before the numbers are seen, and all three use the same deciding metric so the
comparison is fair:

    primary metric  : sensitivity at 70% specificity (the WHO screening operating point)
    tie-break       : AUC
    reported with   : n and a Wilson 95% CI on the sensitivity

The metric here is P2's selection-internal tool and must be consistent for ranking.
Person 4's `p4_stats` is the authority for the numbers that go in the report; a test
checks the two agree to tolerance when both are present.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

TARGET_SPECIFICITY = 0.70


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for k successes in n. Matches p4_stats.wilson_score_interval."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def sensitivity_at_specificity(y, p, target_spec=TARGET_SPECIFICITY):
    """Sensitivity when the threshold is set so specificity = target on these data.

    Threshold = the `target`-quantile of the negative scores (so a `target` fraction of
    negatives fall below it). Returns (sensitivity, threshold, n_pos, n_tp).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    neg = p[y == 0]
    pos = p[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan"), int(len(pos)), 0
    thr = float(np.quantile(neg, target_spec))      # target spec -> fraction of neg below thr
    tp = int((pos >= thr).sum())
    return tp / len(pos), thr, int(len(pos)), tp


def auc(y, p):
    import pandas as pd
    y = np.asarray(y).astype(int)
    r = pd.Series(np.asarray(p, dtype=float)).rank().to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@dataclass
class Candidate:
    model_id: str
    variant: str
    head: str
    clahe: bool
    margin: Optional[float]
    sensitivity: float
    sens_ci_low: float
    sens_ci_high: float
    auc: float
    n_pos: int
    n: int

    def better_than(self, other: "Candidate") -> bool:
        """Primary: higher sensitivity at 70% specificity. Tie (within 0.5 pt): higher AUC."""
        if abs(self.sensitivity - other.sensitivity) > 0.005:
            return self.sensitivity > other.sensitivity
        return self.auc > other.auc


def candidate_table(c08_df, label_fn, target_spec=TARGET_SPECIFICITY) -> List[Candidate]:
    """One Candidate per model_id in a C08 development frame. `label_fn(image_id) -> 0/1`
    supplies the TB label (from C01; on the mock, the id prefix)."""
    rows = []
    for model_id, g in c08_df.groupby("model_id"):
        y = np.array([label_fn(i) for i in g.image_id])
        p = g.p_tb_raw.to_numpy(dtype=float)
        sens, _, n_pos, n_tp = sensitivity_at_specificity(y, p, target_spec)
        lo, hi = wilson_ci(n_tp, n_pos)
        first = g.iloc[0]
        # bracket access, not attribute: a column named "head" is shadowed by Series.head()
        m = first["margin"]
        rows.append(Candidate(
            model_id=model_id, variant=str(first["variant"]), head=str(first["head"]),
            clahe=bool(first["clahe"]),
            margin=None if m is None or (isinstance(m, float) and math.isnan(m)) else round(float(m), 2),
            sensitivity=sens, sens_ci_low=lo, sens_ci_high=hi,
            auc=auc(y, p), n_pos=n_pos, n=len(g)))
    return sorted(rows, key=lambda c: (-c.sensitivity, -c.auc))


def pick_best(cands: List[Candidate], among=None) -> Candidate:
    """The best candidate by the pre-registered rule, optionally within a subset filter."""
    pool = [c for c in cands if (among is None or among(c))]
    if not pool:
        raise ValueError("no candidates match the filter")
    best = pool[0]
    for c in pool[1:]:
        if c.better_than(best):
            best = c
    return best


def select_preprocessing(cands: List[Candidate]):
    """P2-08: among the RAD-DINO binary models, choose CLAHE (D08) and margin (D09)."""
    radino_bin = [c for c in cands if c.head == "binary" and c.model_id.startswith("radino")]
    if not radino_bin:
        raise ValueError("no RAD-DINO binary candidates to select preprocessing from")
    best = pick_best(radino_bin)
    return {"clahe": best.clahe, "margin": best.margin, "variant": best.variant,
            "model_id": best.model_id, "sensitivity": best.sensitivity,
            "sens_ci": [best.sens_ci_low, best.sens_ci_high], "auc": best.auc, "n": best.n}


def select_head(cands: List[Candidate], chosen_variant: str):
    """P2-09: binary vs three-class on the chosen variant. Three-class kept as auxiliary
    if binary wins (methodology)."""
    on_variant = [c for c in cands if c.variant == chosen_variant
                  and c.model_id.startswith("radino")]
    heads = {c.head: c for c in on_variant}
    if "binary" not in heads:
        raise ValueError("no binary candidate on variant %s" % chosen_variant)
    if "three_class" not in heads:
        return {"primary_head": "binary", "auxiliary_head": None,
                "note": "only the binary head was trained on this variant"}
    b, t = heads["binary"], heads["three_class"]
    primary = "binary" if b.better_than(t) else "three_class"
    auxiliary = "three_class" if primary == "binary" else "binary"
    return {"primary_head": primary, "auxiliary_head": auxiliary,
            "binary": {"sensitivity": b.sensitivity, "ci": [b.sens_ci_low, b.sens_ci_high], "n": b.n},
            "three_class": {"sensitivity": t.sensitivity, "ci": [t.sens_ci_low, t.sens_ci_high], "n": t.n}}
