# -*- coding: utf-8 -*-
"""Audit A1: probability calibration of the existing outputs on Vietnam TUNE. Descriptive only.

    py -3 artifacts/probability_calibration_audit_20261101_1100/scripts/calib_audit.py --run
"""
import pandas  # noqa: F401  before torch on Windows
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
R16 = ROOT / "artifacts/clinical_fusion_20261101_0330"
R20 = ROOT / "artifacts/multiencoder_ensemble_20261101_0600"
R22 = ROOT / "artifacts/fusion_on_stacked_head_20261101_0800"
V3_HEAD_SHA = "7f61a082"
sys.path.insert(0, str(R16 / "scripts"))
sys.path.insert(0, str(R20 / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fusion_cv as F16  # noqa: E402
import ensemble as E20   # noqa: E402

N_BOOT = 2000
BOOT_SEED = 20261101
BINS = 10
SLOPE_OK, INTERCEPT_OK, ECE_OK = (0.8, 1.2), (-0.2, 0.2), 0.03


def check_protocol():
    h = hashlib.sha256((RUN / "PROTOCOL.md").read_bytes()).hexdigest()
    if h != (TAB / "protocol_hash.txt").read_text(encoding="utf-8").split()[0]:
        raise SystemExit("PROTOCOL.md changed after hashing")
    return h


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, bins=BINS):
    order = np.argsort(p)
    parts = np.array_split(order, bins)
    n = len(y)
    return float(sum(len(k) / n * abs(p[k].mean() - y[k].mean()) for k in parts if len(k)))


def reliability(y, p, bins=BINS):
    order = np.argsort(p)
    rows = []
    for i, k in enumerate(np.array_split(order, bins)):
        rows.append({"bin": i + 1, "n": int(len(k)), "pred_mean": round(float(p[k].mean()), 4),
                     "observed": round(float(y[k].mean()), 4), "n_pos": int(y[k].sum())})
    return rows


def cox(y, p):
    """Calibration intercept (slope fixed at 1) and slope, unpenalised logistic fits on logit(p)."""
    from sklearn.linear_model import LogisticRegression
    z = logit(p)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(z[:, None], y)
    slope, b0 = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    # intercept at slope 1: fit only an offset (y on z with z as fixed offset)
    lr1 = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, fit_intercept=True)
    # offset model: maximise sum log sigmoid(z + a); 1-D, solve by bisection on the score equation
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        g = float(np.sum(y - sigmoid(z + mid)))
        lo, hi = (mid, hi) if g > 0 else (lo, mid)
    return {"intercept": round((lo + hi) / 2, 4), "slope": round(slope, 4), "slope_fit_intercept": round(b0, 4)}


def boot(y, p, rng, fn, n=N_BOOT):
    v = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if y[k].sum() == 0:
            continue
        v.append(fn(y[k], p[k]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def v3_tune_scores(pids):
    head = ROOT / "artifacts/p2/bundle_v3_candidate/head.npz"      # the candidate file itself
    with np.load(head) as z:
        W, b = z["W"], z["b"]
    with np.load(R22 / "tables/X_noNLM_head.npz") as z:             # must be the R22 weights, unchanged
        if not (np.array_equal(W, z["W"]) and np.array_equal(b, z["b"])):
            raise SystemExit("bundle v3 head differs from R22 head")
    sha = hashlib.sha256(head.read_bytes()).hexdigest()
    if not sha.startswith(V3_HEAD_SHA):
        raise SystemExit("bundle v3 head sha mismatch")
    X = E20.load_cache(R20 / "cache/regional_D2_vie_tune_lungs_g12.npz", pids)
    return np.mean([E20.image_logits(X, W[i], b[i]) for i in range(len(W))], 0), sha


def oof_mean(s, X, y):
    return np.mean([F16.cv_oof("M2", s, X, y, seed)[0] for seed in F16.SEEDS], 0)


def audit(name, y, p, rng, scanner=None):
    r = {"output": name, "n": int(len(y)), "prevalence": round(float(y.mean()), 4),
         "mean_pred": round(float(p.mean()), 4), "auroc": round(F16.auc(y, p), 4),
         "brier": round(brier(y, p), 4), "brier_ref": round(brier(y, np.full_like(p, y.mean())), 4)}
    r["brier_skill"] = round(1 - r["brier"] / r["brier_ref"], 4)
    r["brier_lo"], r["brier_hi"] = [round(v, 4) for v in boot(y, p, rng, brier)]
    r["ece"] = round(ece(y, p), 4)
    r["ece_lo"], r["ece_hi"] = [round(v, 4) for v in boot(y, p, rng, ece)]
    r.update(cox(y, p))
    r["probability_shaped"] = bool(SLOPE_OK[0] <= r["slope"] <= SLOPE_OK[1]
                                   and INTERCEPT_OK[0] <= r["intercept"] <= INTERCEPT_OK[1]
                                   and r["ece"] <= ECE_OK)
    per = []
    if scanner is not None:
        for sc in sorted(set(scanner.tolist())):
            k = scanner == sc
            per.append({"output": name, "scanner": int(sc), "n": int(k.sum()), "n_pos": int(y[k].sum()),
                        "prevalence": round(float(y[k].mean()), 4), "mean_pred": round(float(p[k].mean()), 4),
                        "ece": round(ece(y[k], p[k]), 4), **{"intercept": cox(y[k], p[k])["intercept"]}})
    return r, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    proto = check_protocol()
    pids, y, X, scores, fp, past = F16.load()
    scanner = np.where(X[:, F16.CLIN.index("scan1")] == 1, 1, np.where(X[:, F16.CLIN.index("scan2")] == 1, 2, 3))
    print("TUNE %d, %d positive, fingerprint %s (pinned); scanners %s"
          % (len(y), y.sum(), fp[:16], np.bincount(scanner)[1:].tolist()))
    if not a.run:
        print("DRY. Pass --run.")
        return 0
    lunit = scores["lunit"]
    if lunit.max() > 1.0:
        raise SystemExit("comparator score not on [0,1]: max %.3f" % lunit.max())
    v3_raw, head_sha = v3_tune_scores(pids)
    gate = {"v3_raw_tune_auroc": round(F16.auc(y, v3_raw), 4), "expected": 0.7585,
            "v2_raw_tune_auroc": round(F16.auc(y, scores["ours"]), 4), "expected_v2": 0.7237}
    if abs(gate["v3_raw_tune_auroc"] - 0.7585) > 0.0005 or abs(gate["v2_raw_tune_auroc"] - 0.7237) > 0.0005:
        raise SystemExit("reproduction gate failed: %s" % gate)
    print("gate", gate)
    outputs = [("v2_raw_sigmoid", sigmoid(scores["ours"])),
               ("v2_fused_oof", oof_mean(scores["ours"], X, y)),
               ("v3_fused_oof", oof_mean(v3_raw, X, y)),
               ("lunit_raw", lunit)]
    rng = np.random.default_rng(BOOT_SEED)
    rows, per_rows, rel = [], [], []
    for name, p in outputs:
        r, per = audit(name, y, p, rng, scanner)
        rows.append(r)
        per_rows += per
        rel += [{"output": name, **b} for b in reliability(y, p)]
        print("%-16s AUROC %.4f Brier %.4f (ref %.4f) ECE %.4f [%.4f, %.4f] intercept %+.3f slope %.3f -> %s"
              % (name, r["auroc"], r["brier"], r["brier_ref"], r["ece"], r["ece_lo"], r["ece_hi"],
                 r["intercept"], r["slope"], "probability-shaped" if r["probability_shaped"] else "not a probability"))
    F16.write_csv(TAB / "calibration_summary.csv", rows)
    F16.write_csv(TAB / "calibration_per_scanner.csv", per_rows)
    F16.write_csv(TAB / "reliability_bins.csv", rel)
    (TAB / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": proto, "cohort_fingerprint": fp, "v3_head_sha256": head_sha,
        "reproduction_gate": gate, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "bins": BINS,
        "reading_rule": {"slope": SLOPE_OK, "intercept": INTERCEPT_OK, "ece_max": ECE_OK},
        "seconds": round(time.perf_counter() - t0, 1)}, indent=1), encoding="utf-8")
    print("done in %.1f s" % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
