# -*- coding: utf-8 -*-
"""Task 4, corrected: iterative nullspace projection, because a rank sweep was degenerate.

Why this replaces the rank sweep in task4_representation_diagnostics.py:
    The scanner subspace is derived from a 3-class linear layer, so its centred weight
    matrix is 3 x 768 with rank 2 -- the third singular value came out exactly 0.0.
    `torch.linalg.svd(W, full_matrices=False)` returns only 3 right-singular vectors, so
    asking for rank 8, 16, 32 or 64 silently returned the same 3 rows. Every rank above 2
    produced byte-identical results, which is a bug in the sweep, not a finding.
    That run is kept in representation_diagnostics.csv with this correction recorded.

    Removing more scanner information needs iterative nullspace projection: fit a scanner
    classifier on the current representation, project out its subspace, refit on what is
    left, repeat. Each round removes up to 2 further directions.

Still TUNE-only. The scanner probe is cross-validated and never scores rows used to fit it.
The preregistered rank-3 arm is unchanged and is still the one judged against the
acceptance criteria; everything here is a sensitivity analysis.
"""
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import common as C                                                   # noqa: E402
import task3_subspace_tune as T3                                     # noqa: E402

ROUNDS = 16
CHECK_AT = (0, 1, 2, 3, 4, 6, 8, 12, 16)
N_PERM = 20
SEED = 17
AGG = "max"
T0 = time.perf_counter()


def log(m):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, m), flush=True)


def probe_cv(feats, scanner, seed=SEED, folds=5):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.1, max_iter=2000, random_state=seed))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pred = np.empty_like(scanner)
    for tr, te in cv.split(feats, scanner):
        clf.fit(feats[tr], scanner[tr])
        pred[te] = clf.predict(feats[te])
    return (float(balanced_accuracy_score(scanner, pred)),
            float(f1_score(scanner, pred, average="macro")))


def main():
    import torch
    import pandas as pd

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    tbx_path = C.REG / "regional_tbx11k_lungs_g12_v1.npz"
    tids = C.read_ids(tbx_path)
    pos = {i: k for k, i in enumerate(tids)}
    c = c01[c01.image_id.isin(pos)]
    tr = c[(c.split == "train") & (c.category.isin(C.TB_CATS + ("healthy",)))]
    dv = c[c.split == "development"]
    ytr = np.array([1 if x in C.TB_CATS else 0 for x in tr.category])
    ydv = np.array([1 if x in C.TB_CATS else 0 for x in dv.category])
    want = sorted(set([pos[i] for i in tr.image_id]) | set([pos[i] for i in dv.image_id]))
    X, kept = C.stream_regions(tbx_path, keep_idx=want, device=dev)
    row = {int(g): k for k, g in enumerate(kept)}
    Xtr = X[[row[pos[i]] for i in tr.image_id]].clone()
    Xdv = X[[row[pos[i]] for i in dv.image_id]].clone()
    del X
    torch.cuda.empty_cache()
    heads = [C.train_head(Xtr, ytr, Xdv, ydv, how=AGG, seed=sd)[:2] for sd in C.SEEDS]
    del Xtr, Xdv
    torch.cuda.empty_cache()
    log("frozen TB head ready")

    vn_path = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    vids = C.read_ids(vn_path)
    meta, tune_set, _ = C.vietnam_meta()
    tune_pos = sorted([k for k, p in enumerate(vids) if p in tune_set])
    V, _ = C.stream_regions(vn_path, keep_idx=tune_pos, device=dev)
    tune_ids = [vids[k] for k in tune_pos]
    y = np.array([int(meta[p]["xpert_result"]) for p in tune_ids])
    scan = np.array([int(meta[p]["rad_system"]) for p in tune_ids])

    def summary(tokens):
        with torch.no_grad():
            return torch.cat([tokens[i:i + 256].float().mean(1).cpu()
                              for i in range(0, len(tokens), 256)]).numpy()

    # ---- chance level: the probe itself, on permuted scanner labels ----
    base_feats = summary(V)
    rng = np.random.default_rng(SEED)
    perm = [probe_cv(base_feats, rng.permutation(scan))[0] for _ in range(N_PERM)]
    chance = float(np.mean(perm))
    log("probe chance level over %d label permutations: %.4f (sd %.4f, max %.4f)"
        % (N_PERM, chance, float(np.std(perm)), float(np.max(perm))))

    rows = []
    dims_removed = 0
    for r in range(ROUNDS + 1):
        if r in CHECK_AT:
            feats = summary(V)
            pb, pf = probe_cv(feats, scan)
            tb = np.mean([C.score_head(V, W, b, AGG) for W, b in heads], axis=0)
            auc = roc_auc_score(y, tb)
            lo, hi = C.boot_ci(y, tb, n_boot=2000, seed=SEED)
            rows.append([r, dims_removed, round(pb, 4), round(pf, 4),
                         round(float(auc), 4), round(lo, 4), round(hi, 4),
                         round(pb - chance, 4),
                         "PREREGISTERED EQUIVALENT (rank 3 ~ round 1)" if r == 1 else ""])
            log("  round %-3d dims_removed %-4d scanner bal acc %.4f  TB AUROC %.4f [%.4f, %.4f]"
                % (r, dims_removed, pb, auc, lo, hi))
        if r == ROUNDS:
            break
        B, svals, _ = T3.fit_scanner_subspace(V, scan, 3, dev)
        keep = int((np.asarray(svals) > 1e-6).sum())
        B = B[:max(keep, 1)]
        T3.project_out(V, B)
        dims_removed += B.shape[0]
        torch.cuda.empty_cache()

    with io.open(RUN / "tables/inlp_diagnostics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["inlp_round", "dims_removed", "scanner_balanced_acc", "scanner_macro_f1",
                    "tb_auroc_tune", "tb_auroc_lo", "tb_auroc_hi",
                    "scanner_above_chance", "note"])
        w.writerows(rows)

    json.dump({"rounds": ROUNDS, "checked_at": list(CHECK_AT), "seed": SEED,
               "chance_balanced_accuracy": chance, "n_permutations": N_PERM,
               "permutation_sd": float(np.std(perm)),
               "probe_feature": "mean-pooled 768-d token vector per image",
               "evaluated_on": "Vietnam TUNE only, 5-fold CV",
               "dims_removed_per_round": 2,
               "note": "the rank sweep in representation_diagnostics.csv was degenerate "
                       "above rank 2; this file supersedes its rank>2 rows",
               "report_touched": False},
              io.open(RUN / "tables/inlp_diagnostics.json", "w", encoding="utf-8"), indent=2)
    log("saved INLP diagnostics. REPORT untouched.")


if __name__ == "__main__":
    main()
