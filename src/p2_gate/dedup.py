# -*- coding: utf-8 -*-
"""P2-04: near-duplicate search over C07 embeddings. Produces C17.

This is handshake H1. TBX11K states that each image is a unique individual, but that is
a claim about patients, not about files: the same radiograph can still appear twice, and
a duplicate that straddles a split boundary leaks the test set into training. Person 1
cannot freeze the splits (W2) until this list exists and has been reviewed.

Only cross-split pairs go into C17, because those are the ones that leak. Within-split
duplicates are counted and reported, because they quietly shrink the effective sample
size and the report should say so, but they are not a split-freeze blocker.

The same function backs the mock generator, so the mock cannot drift from the real
search.
"""
import numpy as np

# earlier split keeps the image, later split loses it (P1-07)
SPLIT_ORDER = {"train": 0, "development": 1, "calibration": 2,
               "internal_test": 3, "challenge_test": 4}
C17_COLUMNS = ["image_id_a", "split_a", "image_id_b", "split_b", "cosine", "tau"]


def _unit(emb):
    x = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.allclose(n, 1.0, atol=2e-3):
        raise ValueError("C07 promises L2-normalised embeddings; these are not "
                         "(norms %.4f to %.4f)" % (float(n.min()), float(n.max())))
    return x / np.clip(n, 1e-12, None)


# For a provisional pass over TBX11K's own train/val/test before C01 exists. A leak
# between official train and official val is a leak however Person 1 later subdivides
# them, so this answer is useful early and does not depend on their decisions.
OFFICIAL_ORDER = {"train": 0, "val": 1, "test": 2}


def find_near_duplicates(ids, splits, emb, tau, inclusive=True, block=1024,
                         cross_split_only=True, order=None):
    """Every pair with cosine at or above `tau`, as a DataFrame in C17 column order.

    `inclusive` selects >= tau over > tau. D06 fixes this: it is >=, and a pair landing
    exactly on tau is a candidate rather than a near miss. The distinction is not
    hypothetical, the mock contains such a pair on purpose.

    Blocked rather than one N x N matrix: at 11,200 images that matrix is 500 MB.
    """
    import pandas as pd

    order = SPLIT_ORDER if order is None else order
    ids = np.asarray(ids)
    splits = np.asarray(splits)
    unknown = sorted(set(splits.tolist()) - set(order))
    if unknown:
        raise ValueError("unknown split(s) %s; expected %s" % (unknown, sorted(order)))
    x = _unit(emb)

    rows, within = [], 0
    for start in range(0, len(x), block):
        sim = x[start:start + block] @ x.T
        # float32 matmul lands a pair constructed at exactly tau on 0.9799999, so a bare
        # >= would drop the very boundary case D06 rules on. The tolerance is 1e-6, four
        # orders below the 4-decimal precision the cosine is reported at.
        hit_i, hit_j = (np.where(sim >= tau - 1e-6) if inclusive else np.where(sim > tau))
        for bi, j in zip(hit_i, hit_j):
            i = start + int(bi)
            j = int(j)
            if i >= j:                                  # upper triangle only
                continue
            if splits[i] == splits[j]:
                within += 1
                if cross_split_only:
                    continue
            a, b = (i, j) if order[splits[i]] <= order[splits[j]] else (j, i)
            rows.append({"image_id_a": ids[a], "split_a": splits[a],
                         "image_id_b": ids[b], "split_b": splits[b],
                         "cosine": round(float(sim[bi, j]), 4), "tau": float(tau)})

    df = pd.DataFrame(rows, columns=C17_COLUMNS)
    if len(df):
        df = df.sort_values("cosine", ascending=False).reset_index(drop=True)
    return df, within


def threshold_sweep(ids, splits, emb, taus=(0.95, 0.96, 0.97, 0.98, 0.99), block=1024,
                    order=None):
    """How many cross-split candidates each threshold implies.

    D06 pre-registers this: every candidate is reviewed by hand (P1-07), so a lower tau
    costs review time rather than correctness, while too high a tau silently misses real
    duplicates. The sweep is what turns that trade-off into a number before anyone picks.
    """
    import pandas as pd

    x = _unit(emb)
    splits = np.asarray(splits)
    order = SPLIT_ORDER if order is None else order
    lo = min(taus)
    counts = {t: 0 for t in taus}
    for start in range(0, len(x), block):
        sim = x[start:start + block] @ x.T
        hit_i, hit_j = np.where(sim >= lo)
        for bi, j in zip(hit_i, hit_j):
            i = start + int(bi)
            j = int(j)
            if i >= j or splits[i] == splits[j]:
                continue
            c = float(sim[bi, j])
            for t in taus:
                if c >= t:
                    counts[t] += 1
    return pd.DataFrame({"tau": list(taus),
                         "cross_split_candidates": [counts[t] for t in taus]})


def load_c07(path):
    """(ids, emb) from a C07 npz, checking it is what it claims to be."""
    with np.load(path) as z:
        missing = {"image_id", "emb", "l2_normalised"} - set(z.files)
        if missing:
            raise ValueError("%s is not a C07 file; missing %s" % (path, sorted(missing)))
        if not bool(z["l2_normalised"]):
            raise ValueError("%s declares l2_normalised=False" % path)
        variant = str(z["variant"]) if "variant" in z.files else "?"
        if not variant.startswith("full"):
            raise ValueError(
                "dedup runs on the FULL-image variant, not %r. A lung crop throws away "
                "the borders and markers that make two scans of one patient "
                "distinguishable." % variant)
        return z["image_id"].tolist(), z["emb"].copy(), variant
