# -*- coding: utf-8 -*-
"""Task 4: did the projection actually remove scanner information, and at what cost to TB?

Everything here is TUNE-only and cross-validated. The scanner probe is never evaluated on
rows used to fit it.

Design note that decides how this is read: the preregistered rank is 3, but the centred
class-weight matrix of a 3-class problem has rank 2 (the third singular value came out
exactly 0.0 in task3). So the preregistered projection removes 2 real scanner directions
plus one arbitrary direction from the null space. A rank sweep is therefore run as a
TUNE-only sensitivity analysis, to establish whether the preregistered rank was simply too
small to remove the information. The preregistered rank-3 arm remains the one judged
against the acceptance criteria; the sweep does not change it.
"""
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

RANKS = [2, 3, 8, 16, 32, 64]          # 3 is the preregistered value
N_PERM = 200
SEED = 17
AGG = "max"
T0 = time.perf_counter()


def log(m):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, m), flush=True)


def probe_cv(feats, scanner, seed=SEED, folds=5):
    """Cross-validated scanner prediction. Never scores rows used to fit."""
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

    # ---- TBX11K head (identical procedure to task3) ----
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    tbx_path = C.REG / "regional_tbx11k_lungs_g12_v1.npz"
    tids = C.read_ids(tbx_path)
    pos = {i: k for k, i in enumerate(tids)}
    c = c01[c01.image_id.isin(pos)]
    tr = c[(c.split == "train") & (c.category.isin(C.TB_CATS + ("healthy",)))]
    dv = c[c.split == "development"]
    ytr = np.array([1 if x in C.TB_CATS else 0 for x in tr.category])
    ydv = np.array([1 if x in C.TB_CATS else 0 for x in dv.category])
    tr_idx = [pos[i] for i in tr.image_id]
    dv_idx = [pos[i] for i in dv.image_id]
    want = sorted(set(tr_idx) | set(dv_idx))
    X, kept = C.stream_regions(tbx_path, keep_idx=want, device=dev)
    row = {int(g): k for k, g in enumerate(kept)}
    Xtr = X[[row[i] for i in tr_idx]].clone()
    Xdv = X[[row[i] for i in dv_idx]].clone()
    del X
    torch.cuda.empty_cache()

    heads = [C.train_head(Xtr, ytr, Xdv, ydv, how=AGG, seed=sd)[:2] for sd in C.SEEDS]
    log("frozen TB head trained (TBX11K dev %.4f)"
        % roc_auc_score(ydv, np.mean([C.score_head(Xdv, W, b, AGG) for W, b in heads], 0)))

    # ---- Vietnam TUNE tokens ----
    vn_path = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    vids = C.read_ids(vn_path)
    meta, tune_set, _ = C.vietnam_meta()
    tune_pos = [k for k, p in enumerate(vids) if p in tune_set]
    V0, _ = C.stream_regions(vn_path, keep_idx=tune_pos, device=dev)
    tune_ids = [vids[k] for k in sorted(tune_pos)]
    y = np.array([int(meta[p]["xpert_result"]) for p in tune_ids])
    scan = np.array([int(meta[p]["rad_system"]) for p in tune_ids])
    log("Vietnam TUNE tokens on device: %s  (%d TB)" % (tuple(V0.shape), y.sum()))

    def summary_feats(tokens):
        """Mean-pooled token vector per image: the fixed summary the probe reads."""
        with torch.no_grad():
            return torch.cat([tokens[i:i + 256].float().mean(1).cpu()
                              for i in range(0, len(tokens), 256)]).numpy()

    base_feats = summary_feats(V0)
    base_probe_bal, base_probe_f1 = probe_cv(base_feats, scan)
    base_tb = np.mean([C.score_head(V0, W, b, AGG) for W, b in heads], axis=0)
    base_auc = roc_auc_score(y, base_tb)
    log("BEFORE projection: scanner balanced acc %.4f | TB AUROC %.4f"
        % (base_probe_bal, base_auc))

    # permutation reference for the probe, fixed seed
    rng = np.random.default_rng(SEED)
    perm_scores = []
    for _ in range(min(N_PERM, 200)):
        perm_scores.append(balanced_accuracy_score(
            rng.permutation(scan), scan))      # label-shuffle reference for chance level
    chance = float(np.mean(perm_scores))
    log("  chance balanced accuracy under %d label permutations: %.4f" % (N_PERM, chance))

    rows = []
    rows.append(["none (baseline)", 0, round(base_probe_bal, 4), round(base_probe_f1, 4),
                 round(base_auc, 4), 0.0, 0.0, ""])

    for rank in RANKS:
        Vp = V0.clone()
        B, svals, _ = T3.fit_scanner_subspace(V0, scan, rank, dev)
        T3.project_out(Vp, B)
        feats = summary_feats(Vp)
        pb, pf = probe_cv(feats, scan)
        tb = np.mean([C.score_head(Vp, W, b, AGG) for W, b in heads], axis=0)
        auc = roc_auc_score(y, tb)
        n_zero = int((np.asarray(svals)[:rank] < 1e-6).sum()) if len(svals) >= rank else 0
        rows.append(["scanner subspace", rank, round(pb, 4), round(pf, 4), round(auc, 4),
                     round(pb - base_probe_bal, 4), round(auc - base_auc, 4),
                     "PREREGISTERED" if rank == 3 else ""])
        log("  rank %-3d  scanner bal acc %.4f (%+0.4f)  TB AUROC %.4f (%+0.4f)  %s"
            % (rank, pb, pb - base_probe_bal, auc, auc - base_auc,
               "<- preregistered" if rank == 3 else ""))
        del Vp, B
        torch.cuda.empty_cache()

    import csv
    with io.open(RUN / "tables/representation_diagnostics.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["transform", "rank", "scanner_balanced_acc", "scanner_macro_f1",
                    "tb_auroc_tune", "delta_scanner_balacc", "delta_tb_auroc", "note"])
        w.writerows(rows)

    json.dump({"chance_balanced_accuracy": chance, "n_permutations": N_PERM,
               "seed": SEED, "folds": 5, "ranks_tested": RANKS,
               "preregistered_rank": 3,
               "probe_feature": "mean-pooled 768-d token vector per image",
               "baseline": {"scanner_balanced_acc": base_probe_bal,
                            "scanner_macro_f1": base_probe_f1,
                            "tb_auroc_tune": float(base_auc)},
               "evaluated_on": "Vietnam TUNE only, 5-fold CV, never scoring fitted rows",
               "report_touched": False},
              io.open(RUN / "tables/representation_diagnostics.json", "w", encoding="utf-8"),
              indent=2)
    log("saved representation diagnostics. REPORT untouched.")


if __name__ == "__main__":
    main()
