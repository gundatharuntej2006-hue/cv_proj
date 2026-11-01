# -*- coding: utf-8 -*-
"""Deeper uncertainty and reproducibility checks on the experiments already run.

No new method, no new selection, no new threshold. Three checks:

  A. Determinism: re-derive the frozen scores and confirm they are bit-identical.
  B. Subspace seed sensitivity: does the null in SUBSPACE_RESULTS.md depend on seed 17?
  C. Better uncertainty for the offset arm: scanner-stratified bootstrap, and a
     permutation test asking whether +0.0206 is larger than random per-group offsets give.

C re-estimates uncertainty for a point estimate that was already computed under the frozen
manifest. It does not change the estimate, the offsets or any threshold.
"""
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import common as C                                                   # noqa: E402

N_BOOT = 10000
N_PERM = 2000
SEED = 0
T0 = time.perf_counter()


def log(m):
    print("[%6.1fs] %s" % (time.perf_counter() - T0, m), flush=True)


def load_scores():
    rows = list(csv.DictReader(io.open(
        RUN / "tables/vietnam_case_scores.csv", encoding="utf-8-sig")))
    return (np.array([int(r["xpert_result"]) for r in rows]),
            np.array([float(r["score_baseline"]) for r in rows]),
            np.array([float(r["score_P1_projected_frozen_head"]) for r in rows]),
            np.array([int(r["rad_system"]) for r in rows]),
            np.array([r["split_half"] for r in rows]))


def check_a_determinism():
    """Re-run the head training and scoring; the result must be bit-identical."""
    import torch
    import pandas as pd
    import task3_subspace_tune as T3                                  # noqa: F401

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    tbx = C.REG / "regional_tbx11k_lungs_g12_v1.npz"
    tids = C.read_ids(tbx)
    pos = {i: k for k, i in enumerate(tids)}
    c = c01[c01.image_id.isin(pos)]
    tr = c[(c.split == "train") & (c.category.isin(C.TB_CATS + ("healthy",)))]
    dv = c[c.split == "development"]
    ytr = np.array([1 if x in C.TB_CATS else 0 for x in tr.category])
    ydv = np.array([1 if x in C.TB_CATS else 0 for x in dv.category])
    want = sorted(set([pos[i] for i in tr.image_id]) | set([pos[i] for i in dv.image_id]))
    X, kept = C.stream_regions(tbx, keep_idx=want, device=dev, verbose=False)
    row = {int(g): k for k, g in enumerate(kept)}
    Xtr = X[[row[pos[i]] for i in tr.image_id]].clone()
    Xdv = X[[row[pos[i]] for i in dv.image_id]].clone()
    del X
    torch.cuda.empty_cache()
    heads = [C.train_head(Xtr, ytr, Xdv, ydv, how="max", seed=sd)[:2] for sd in C.SEEDS]
    del Xtr, Xdv
    torch.cuda.empty_cache()

    vn = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    V, _ = C.stream_regions(vn, device=dev, verbose=False)
    got = np.mean([C.score_head(V, W, b, "max") for W, b in heads], axis=0)
    del V
    torch.cuda.empty_cache()

    _, base, _, _, _ = load_scores()
    exact = int((np.abs(got - base) < 5e-7).sum())
    maxd = float(np.abs(got - base).max())
    log("A. determinism: %d / %d scores reproduce to 5e-7, max deviation %.3e"
        % (exact, len(base), maxd))
    return {"n": len(base), "n_matching_5e-7": exact, "max_abs_deviation": maxd,
            "bit_identical": bool(maxd == 0.0)}, heads


def check_b_seed_sensitivity(heads):
    """Does the subspace null depend on the seed used to fit the subspace?"""
    import torch
    import task3_subspace_tune as T3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vn = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    vids = C.read_ids(vn)
    meta, tune_set, _ = C.vietnam_meta()
    tune_pos = sorted([k for k, p in enumerate(vids) if p in tune_set])
    V0, _ = C.stream_regions(vn, keep_idx=tune_pos, device=dev, verbose=False)
    ids = [vids[k] for k in tune_pos]
    y = np.array([int(meta[p]["xpert_result"]) for p in ids])
    scan = np.array([int(meta[p]["rad_system"]) for p in ids])

    base = roc_auc_score(y, np.mean([C.score_head(V0, W, b, "max") for W, b in heads], 0))
    out = []
    for sd in (17, 29, 43, 61):
        T3.SUBSPACE_SEED = sd
        Vp = V0.clone()
        B, svals, _ = T3.fit_scanner_subspace(V0, scan, 3, dev)
        T3.project_out(Vp, B)
        a = roc_auc_score(y, np.mean([C.score_head(Vp, W, b, "max") for W, b in heads], 0))
        out.append({"subspace_seed": sd, "tune_auroc": round(float(a), 6),
                    "delta_vs_baseline": round(float(a - base), 6),
                    "singular_values": [round(float(x), 3) for x in svals]})
        log("B. subspace seed %2d -> TUNE AUROC %.6f (%+0.6f)" % (sd, a, a - base))
        del Vp, B
        torch.cuda.empty_cache()
    T3.SUBSPACE_SEED = 17
    del V0
    torch.cuda.empty_cache()
    d = [r["delta_vs_baseline"] for r in out]
    return {"baseline_tune_auroc": round(float(base), 6), "per_seed": out,
            "delta_range": [min(d), max(d)],
            "conclusion": "the null does not depend on the subspace seed"
                          if max(abs(x) for x in d) < 0.005 else "seed-dependent"}


def check_c_offset_uncertainty():
    """Scanner-stratified bootstrap and a permutation test for the offset arm."""
    y, base, _, scan, half = load_scores()
    P = json.load(io.open(RUN / "tables/frozen_parameters.json", encoding="utf-8"))
    off = P["scanner_offsets_tune_non_tb_median"]
    rep = half == "REPORT"
    yr, sr, gr = y[rep], base[rep], scan[rep]
    cr = sr.copy()
    for g in (1, 2, 3):
        cr[gr == g] -= off[str(g)]
    point = roc_auc_score(yr, cr) - roc_auc_score(yr, sr)

    # scanner-stratified paired bootstrap: resample within each scanner, preserving the
    # group sizes that the offset correction operates on
    rng = np.random.default_rng(SEED)
    idx_by_g = {g: np.where(gr == g)[0] for g in (1, 2, 3)}
    d = []
    for _ in range(N_BOOT):
        k = np.concatenate([rng.choice(v, len(v), replace=True) for v in idx_by_g.values()])
        if len(np.unique(yr[k])) < 2:
            continue
        d.append(roc_auc_score(yr[k], cr[k]) - roc_auc_score(yr[k], sr[k]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    log("C1. stratified bootstrap (%d): delta %+0.4f, CI [%+0.4f, %+0.4f], %s"
        % (len(d), point, lo, hi, "excludes 0" if lo > 0 or hi < 0 else "includes 0"))

    # permutation reference: random per-group offsets of the same magnitudes
    mags = np.array([off["1"], off["2"], off["3"]])
    null = []
    for _ in range(N_PERM):
        perm = rng.permutation(mags)
        z = sr.copy()
        for j, g in enumerate((1, 2, 3)):
            z[gr == g] -= perm[j]
        null.append(roc_auc_score(yr, z) - roc_auc_score(yr, sr))
    null = np.array(null)
    pval = float((null >= point).mean())
    log("C2. permutation of the same three offsets across scanners (%d draws): "
        "observed %+0.4f, null mean %+0.4f, p = %.4f" % (N_PERM, point, null.mean(), pval))

    return {"point_estimate": round(float(point), 6),
            "stratified_bootstrap": {"n": int(len(d)), "ci95": [round(float(lo), 4),
                                                                round(float(hi), 4)],
                                     "excludes_zero": bool(lo > 0 or hi < 0)},
            "permutation_test": {"n": N_PERM, "null_mean": round(float(null.mean()), 6),
                                 "null_max": round(float(null.max()), 6),
                                 "p_value": pval,
                                 "note": "null permutes the SAME three offsets across the "
                                         "three scanners, so it tests whether the assignment "
                                         "matters, not whether any shift helps"}}


def main():
    res = {}
    res["A_determinism"], heads = check_a_determinism()
    res["B_subspace_seed_sensitivity"] = check_b_seed_sensitivity(heads)
    res["C_offset_uncertainty"] = check_c_offset_uncertainty()
    res["note"] = ("deeper checks on experiments already frozen and scored; no parameter, "
                   "threshold or acceptance criterion was changed")
    json.dump(res, io.open(RUN / "tables/robustness_checks.json", "w", encoding="utf-8"),
              indent=2)
    log("saved tables/robustness_checks.json")


if __name__ == "__main__":
    main()
