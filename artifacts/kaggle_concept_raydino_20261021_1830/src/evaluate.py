# -*- coding: utf-8 -*-
"""Matched downstream head, metrics, paired bootstrap and threshold discipline.

Every threshold and calibrator is fitted on the designated development partition only.
`fit_threshold` refuses any partition name that is not the declared development one.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DEV_PARTITION = "development"


# ----------------------------------------------------------------- matched head
def fit_head(x_train, y_train, c=1.0, seed=17, max_iter=4000):
    """The matched downstream head: standardise + regularised logistic regression.

    Identical for both encoders. The only thing that differs between arms is the input
    feature matrix, which is the point of the comparison. Returns the fitted pipeline so
    the weights can be stored, not just the seed.
    """
    if len(set(np.asarray(y_train).tolist())) < 2:
        raise ValueError("training labels contain a single class")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=c, max_iter=max_iter, random_state=seed))
    clf.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train).astype(int))
    return clf


def select_C(x_tr, y_tr, x_dev, y_dev, grid=(0.01, 0.1, 1.0, 10.0), seed=17):
    """Pick regularisation on the DEVELOPMENT split only. Small, declared grid."""
    best = None
    for c in grid:
        clf = fit_head(x_tr, y_tr, c=c, seed=seed)
        a = roc_auc_score(y_dev, clf.predict_proba(np.asarray(x_dev, np.float32))[:, 1])
        if best is None or a > best[1]:
            best = (c, float(a))
    return {"C": best[0], "development_auroc": best[1], "grid": list(grid)}


def score(clf, x):
    return clf.predict_proba(np.asarray(x, dtype=np.float32))[:, 1]


# ----------------------------------------------------------------- thresholds
def fit_threshold(scores, labels, partition_name, target_specificity=0.70):
    """Threshold at a target specificity, computed on the development partition ONLY."""
    if partition_name != DEV_PARTITION:
        raise ValueError(
            "thresholds may only be fitted on %r, got %r. Fitting a threshold on any "
            "evaluation partition invalidates the comparison."
            % (DEV_PARTITION, partition_name))
    s = np.asarray(scores, float)
    y = np.asarray(labels).astype(int)
    if (y == 0).sum() < 10:
        raise ValueError("too few negatives (%d) to set a threshold" % (y == 0).sum())
    return float(np.quantile(s[y == 0], target_specificity))


def sens_at_spec(scores, labels, spec):
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return float("nan")
    t = np.quantile(s[y == 0], spec)
    return float((s[y == 1] >= t).mean())


def spec_at_sens(scores, labels, sens):
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return float("nan")
    t = np.quantile(s[y == 1], 1.0 - sens)
    return float((s[y == 0] < t).mean())


def operating_point(scores, labels, thr):
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    f = s >= thr
    tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum()); tn = int((~f & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1) if tp + fp else None,
            "referral_rate": (tp + fp) / max(len(y), 1)}


# ----------------------------------------------------------------- uncertainty
def boot_ci(y, s, n_boot=4000, seed=0, groups=None):
    """Bootstrap CI for AUROC. If `groups` (patient ids) is given, resample by patient."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    if groups is None:
        idx_pool = [np.array([i]) for i in range(len(y))]
    else:
        g = {}
        for i, gid in enumerate(groups):
            g.setdefault(gid, []).append(i)
        idx_pool = [np.array(v) for v in g.values()]
    v = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_pool), len(idx_pool))
        k = np.concatenate([idx_pool[p] for p in pick])
        if len(np.unique(y[k])) < 2:
            continue
        v.append(roc_auc_score(y[k], s[k]))
    if not v:
        return (float("nan"), float("nan"))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_boot(y, s_a, s_b, n_boot=4000, seed=0, groups=None):
    """CI for AUROC(b) - AUROC(a) on identical cases. Resamples by patient if given."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    if groups is None:
        idx_pool = [np.array([i]) for i in range(len(y))]
    else:
        g = {}
        for i, gid in enumerate(groups):
            g.setdefault(gid, []).append(i)
        idx_pool = [np.array(v) for v in g.values()]
    d = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_pool), len(idx_pool))
        k = np.concatenate([idx_pool[p] for p in pick])
        if len(np.unique(y[k])) < 2:
            continue
        d.append(roc_auc_score(y[k], s_b[k]) - roc_auc_score(y[k], s_a[k]))
    if not d:
        return {"delta": float("nan"), "ci95": [float("nan")] * 2,
                "excludes_zero": False, "n_resamples": 0}
    d = np.array(d)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta": float(roc_auc_score(y, s_b) - roc_auc_score(y, s_a)),
            "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "n_resamples": int(len(d))}


def bh_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns (adjusted p, rejected) in the input order."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], bool)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out, out <= alpha


def metrics_block(y, s, label, groups=None, seed=0):
    y = np.asarray(y).astype(int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"group": label, "n": int(len(y)), "n_pos": int(y.sum()),
                "auroc": None, "note": "fewer than two classes present"}
    lo, hi = boot_ci(y, s, seed=seed, groups=groups)
    return {"group": label, "n": int(len(y)), "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, s)), "auroc_ci95": [lo, hi],
            "auprc": float(average_precision_score(y, s)),
            "auprc_baseline": float(y.mean()),
            "patient_level_bootstrap": groups is not None}
