# -*- coding: utf-8 -*-
"""Task 3, TUNE phase: fit the scanner subspace and evaluate on TUNE only.

Implements PRIMARY_EXPERIMENT_PREREGISTRATION.md section 4.

  * The TB head is trained on TBX11K train only (healthy-only negatives, lungs-only
    masking, 12x12 regions, max aggregation, seeds 17/29/43/61/79), early-stopped on
    TBX11K development. Vietnam TB labels never enter head training.
  * The scanner subspace is fitted on Vietnam TUNE tokens using rad_system labels only.
    No TB label is used to fit the transform.
  * Rank is fixed at 3, declared before any result was seen.

Per-case scores are computed for the whole Vietnam cohort because scoring is a
deterministic function of a frozen pipeline, but **no REPORT metric is computed, printed
or saved by this script**. REPORT metrics come from task_report_sealed.py, which runs only
after FROZEN_MANIFEST.md is written.
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

RANK = 3                       # predeclared, section 4
SUBSPACE_SEED = 17
SUBSPACE_EPOCHS = 60
SUBSPACE_LR = 1e-2
SUBSPACE_BS = 4096
AGG = "max"

# Published values this run must reproduce before anything new is believed.
EXPECT_TBX11K_DEV = 0.9368
EXPECT_VIETNAM_TUNE = 0.7237


def log(msg):
    print("[%7.1fs] %s" % (time.perf_counter() - T0, msg), flush=True)


T0 = time.perf_counter()


def fit_scanner_subspace(tokens, labels, rank, device):
    """Multinomial logistic regression on token vectors -> orthonormal rank-r basis.

    tokens : [n, R, C] float16 device tensor (Vietnam TUNE only)
    labels : [n] int array of rad_system, mapped to 0..K-1
    Returns B : [rank, C] float32, orthonormal rows spanning the scanner subspace.
    """
    import torch

    n, regions, chan = tokens.shape
    classes = sorted(set(int(x) for x in labels))
    remap = {c: i for i, c in enumerate(classes)}
    y_img = torch.tensor([remap[int(x)] for x in labels], device=device)
    # each region of an image inherits the image's scanner label
    torch.manual_seed(SUBSPACE_SEED)
    lin = torch.nn.Linear(chan, len(classes)).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=SUBSPACE_LR, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()

    idx = torch.arange(n, device=device)
    for ep in range(SUBSPACE_EPOCHS):
        perm = idx[torch.randperm(n, device=device)]
        tot, seen = 0.0, 0
        for i in range(0, n, 64):
            k = perm[i:i + 64]
            x = tokens[k].float().reshape(-1, chan)
            t = y_img[k].repeat_interleave(regions)
            opt.zero_grad()
            loss = lossf(lin(x), t)
            loss.backward()
            opt.step()
            tot += float(loss) * len(k)
            seen += len(k)
        if ep % 15 == 0 or ep == SUBSPACE_EPOCHS - 1:
            log("    subspace epoch %3d  token cross-entropy %.4f" % (ep, tot / seen))

    W = lin.weight.detach().float()                       # [K, C]
    W = W - W.mean(0, keepdim=True)                       # drop the common direction
    # rank-r orthonormal basis of the row space, by SVD (stable for near-collinear rows)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    B = Vh[:rank].contiguous()                            # [rank, C], orthonormal rows
    return B, S.cpu().numpy(), classes


def project_out(tokens, B, chunk=256):
    """x <- x - B^T (B x), in place, block by block to bound VRAM."""
    import torch
    with torch.no_grad():
        for i in range(0, len(tokens), chunk):
            g = tokens[i:i + chunk].float()
            coef = g @ B.T                                # [m, R, rank]
            g = g - coef @ B
            tokens[i:i + chunk] = g.half()
    return tokens


def main():
    import torch
    import pandas as pd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device %s" % dev)

    # ---------------------------------------------------- TBX11K train / development
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    tbx_path = C.REG / "regional_tbx11k_lungs_g12_v1.npz"
    tids = C.read_ids(tbx_path)
    pos = {i: k for k, i in enumerate(tids)}
    c = c01[c01.image_id.isin(pos)]
    tr = c[(c.split == "train") & (c.category.isin(C.TB_CATS + ("healthy",)))]
    dv = c[c.split == "development"]
    tr_idx = [pos[i] for i in tr.image_id]
    dv_idx = [pos[i] for i in dv.image_id]
    ytr = np.array([1 if x in C.TB_CATS else 0 for x in tr.category])
    ydv = np.array([1 if x in C.TB_CATS else 0 for x in dv.category])
    log("TBX11K train %d (%d TB) / development %d (%d TB)"
        % (len(tr_idx), ytr.sum(), len(dv_idx), ydv.sum()))

    want = sorted(set(tr_idx) | set(dv_idx))
    X, kept = C.stream_regions(tbx_path, keep_idx=want, device=dev)
    row = {int(g): k for k, g in enumerate(kept)}
    Xtr = X[[row[i] for i in tr_idx]]
    Xdv = X[[row[i] for i in dv_idx]]
    del X
    torch.cuda.empty_cache()
    log("VRAM after TBX11K: %.2f GB allocated" % (torch.cuda.memory_allocated() / 1e9))

    # ---------------------------------------------------- train the frozen TB head
    heads = []
    for sd in C.SEEDS:
        W, b, best = C.train_head(Xtr, ytr, Xdv, ydv, how=AGG, seed=sd)
        heads.append((W, b))
        log("  head seed %2d  best TBX11K dev AUROC %.4f" % (sd, best))
    dev_scores = np.mean([C.score_head(Xdv, W, b, AGG) for W, b in heads], axis=0)
    dev_auc = roc_auc_score(ydv, dev_scores)
    log("TBX11K development AUROC (5-seed ensemble): %.4f   [published %.4f]"
        % (dev_auc, EXPECT_TBX11K_DEV))

    # ---------------------------------------------------- Vietnam
    vn_path = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    vids = C.read_ids(vn_path)
    meta, tune_set, report_set = C.vietnam_meta()
    V, _ = C.stream_regions(vn_path, device=dev)
    log("VRAM after Vietnam: %.2f GB allocated" % (torch.cuda.memory_allocated() / 1e9))

    y_vn = np.array([int(meta[p]["xpert_result"]) for p in vids])
    scan = np.array([int(meta[p]["rad_system"]) for p in vids])
    is_tune = np.array([p in tune_set for p in vids])
    assert is_tune.sum() == 1352 and (~is_tune).sum() == 1356

    base_scores = np.mean([C.score_head(V, W, b, AGG) for W, b in heads], axis=0)
    tune_auc = roc_auc_score(y_vn[is_tune], base_scores[is_tune])
    log("Vietnam TUNE AUROC, unprojected: %.4f   [published %.4f]"
        % (tune_auc, EXPECT_VIETNAM_TUNE))

    ok = abs(tune_auc - EXPECT_VIETNAM_TUNE) < 0.01 and abs(dev_auc - EXPECT_TBX11K_DEV) < 0.01
    log("END-TO-END EQUIVALENCE CHECK: %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit("streaming pipeline does not reproduce the published baseline; "
                         "refusing to report any derived result")

    # ---------------------------------------------------- fit the scanner subspace on TUNE
    log("fitting scanner subspace on Vietnam TUNE tokens (rank %d, seed %d)"
        % (RANK, SUBSPACE_SEED))
    tune_rows = np.where(is_tune)[0]
    B, svals, classes = fit_scanner_subspace(V[tune_rows], scan[tune_rows], RANK, dev)
    log("  singular values of the centred class-weight matrix: %s"
        % np.round(svals, 4).tolist())
    orth = float((B @ B.T - torch.eye(RANK, device=dev)).abs().max())
    log("  orthonormality residual: %.2e" % orth)

    # ---------------------------------------------------- Arm P1: frozen head, project at inference
    Vp = V                                        # projected in place; V is not needed after
    project_out(Vp, B)
    torch.cuda.empty_cache()
    p1_scores = np.mean([C.score_head(Vp, W, b, AGG) for W, b in heads], axis=0)
    p1_tune = roc_auc_score(y_vn[is_tune], p1_scores[is_tune])
    log("ARM P1 (preregistered: head frozen, project at inference) TUNE AUROC %.4f" % p1_tune)

    # ---------------------------------------------------- Arm P2: retrain head on projected TBX11K
    project_out(Xtr, B)
    project_out(Xdv, B)
    torch.cuda.empty_cache()
    heads2 = []
    for sd in C.SEEDS:
        W, b, best = C.train_head(Xtr, ytr, Xdv, ydv, how=AGG, seed=sd)
        heads2.append((W, b))
        log("  P2 head seed %2d  best TBX11K dev AUROC %.4f" % (sd, best))
    dev2 = roc_auc_score(ydv, np.mean([C.score_head(Xdv, W, b, AGG) for W, b in heads2], axis=0))
    p2_scores = np.mean([C.score_head(Vp, W, b, AGG) for W, b in heads2], axis=0)
    p2_tune = roc_auc_score(y_vn[is_tune], p2_scores[is_tune])
    log("ARM P2 (exploratory: head retrained in the projected space) TUNE AUROC %.4f "
        "(TBX11K dev %.4f)" % (p2_tune, dev2))

    # ---------------------------------------------------- save everything
    np.save(RUN / "tables/scanner_subspace_basis.npy", B.cpu().numpy())
    with io.open(RUN / "tables/vietnam_case_scores.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "xpert_result", "rad_system", "split_half",
                    "score_baseline", "score_P1_projected_frozen_head",
                    "score_P2_projected_retrained_head"])
        for k, p in enumerate(vids):
            w.writerow([p, int(y_vn[k]), int(scan[k]),
                        "TUNE" if is_tune[k] else "REPORT",
                        "%.6f" % base_scores[k], "%.6f" % p1_scores[k],
                        "%.6f" % p2_scores[k]])

    out = {
        "agg": AGG, "grid": C.GRID, "seeds": list(C.SEEDS), "rank": RANK,
        "subspace": {"seed": SUBSPACE_SEED, "epochs": SUBSPACE_EPOCHS, "lr": SUBSPACE_LR,
                     "classes": classes, "singular_values": [float(x) for x in svals],
                     "orthonormality_residual": orth,
                     "fitted_on": "Vietnam TUNE tokens only, rad_system labels only",
                     "tb_labels_used_in_transform": False},
        "equivalence_check": {"tbx11k_dev_auroc": float(dev_auc),
                              "tbx11k_dev_published": EXPECT_TBX11K_DEV,
                              "vietnam_tune_auroc_unprojected": float(tune_auc),
                              "vietnam_tune_published": EXPECT_VIETNAM_TUNE,
                              "passed": bool(ok)},
        "tune_results": {"baseline": float(tune_auc), "P1": float(p1_tune),
                         "P2": float(p2_tune), "P2_tbx11k_dev": float(dev2)},
        "report_metrics_computed": False,
        "elapsed_s": round(time.perf_counter() - T0, 1),
    }
    json.dump(out, io.open(RUN / "tables/subspace_tune.json", "w", encoding="utf-8"), indent=2)
    log("saved. REPORT metrics NOT computed in this script.")


if __name__ == "__main__":
    main()
