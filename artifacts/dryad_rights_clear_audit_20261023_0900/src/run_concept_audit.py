# -*- coding: utf-8 -*-
"""Phase 3: TUNE-only CLEAR concept audit. PREPARED AND RUNNABLE, NOT YET EXECUTED.

Blocked locally on host RAM (0.8 GB free of 16.4; CLEAR needs a 603 MB checkpoint plus a
slice of a 1,131 MB concept-embedding tensor). Runs unchanged on Kaggle or on this machine
once memory frees.

Hard guarantees, enforced in code and covered by tests:
  * Vietnam **TUNE only**. REPORT pids are never loaded; a guard raises if one appears.
  * TBX11K `internal_test` is never referenced.
  * Only the **9 exact** vocabulary entries in configs/concept_indices.json are used.
    The 3 concepts with no exact entry are EXCLUDED, not approximated.
  * No concept score is ever subtracted from, or used to modify, a TB score.
  * Nothing is written outside the run directory, and no per-case table leaves it.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

HF_REPO = "peterhan91/CLEAR"
HF_REV = "142f169c8b006b0f28cf55c6a237c3003db5ed8f"      # pinned; no floating main
CLEAR_INPUT_PX = 448
VN_META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
VN_SPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VN_SCORES = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"
IMG_DIR = ROOT / "data/processed/v1/vietnam_img512"
N_BOOT = 10000
SEED = 20261023


# --------------------------------------------------------------------- guards
class SplitViolation(RuntimeError):
    pass


def load_tune_only():
    """Return the TUNE cohort. Raises if a REPORT pid would be included."""
    split = json.load(io.open(VN_SPLIT, encoding="utf-8"))
    tune = set(str(x) for x in split["tune"])
    report = set(str(x) for x in split["report"])
    if tune & report:
        raise SplitViolation("TUNE and REPORT overlap in the split manifest")

    meta = {str(r["pid"]): r for r in csv.DictReader(io.open(VN_META, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(io.open(VN_SCORES, encoding="utf-8-sig")))
    sel = [r for r in rows if str(r["pid"]) in tune]

    leaked = [r["pid"] for r in sel if str(r["pid"]) in report]
    if leaked:
        raise SplitViolation("REPORT pids reached the analysis set: %s" % leaked[:3])
    if len(sel) != len(tune):
        raise SplitViolation("expected %d TUNE rows, got %d" % (len(tune), len(sel)))
    ids = [str(r["pid"]) for r in sel]
    if len(set(ids)) != len(ids):
        raise SplitViolation("duplicate pid in the TUNE set")

    return {
        "pid": ids,
        "y": np.array([int(meta[p]["xpert_result"]) for p in ids]),
        "past_tb": np.array([int(meta[p]["past_tb"]) for p in ids]),
        "rad_system": np.array([int(meta[p]["rad_system"]) for p in ids]),
        "age_group": np.array([int(meta[p]["age_group"]) for p in ids]),
        "sex": np.array([int(meta[p]["sex"]) for p in ids]),
        "symptoms": np.array([sum(int(meta[p][k]) for k in ("c", "f", "wl", "ns"))
                              for p in ids]),
        "clf_score": np.array([float(r["score_baseline"]) for r in sel]),
    }


def concept_map():
    m = json.load(io.open(RUN / "configs/concept_indices.json", encoding="utf-8"))
    primary = [c for c in m["mapping"] if c["concept_idx"] is not None]
    if len(primary) != m["n_primary"]:
        raise AssertionError("concept map is inconsistent with its own count")
    return m, primary


# --------------------------------------------------------------------- inference
def encode_concepts(pids, device="cuda", batch=16, log=print):
    """Encode TUNE images with CLEAR and return [n, 9] similarities to the pinned concepts.

    Requires the pinned CLEAR checkpoint and the concept-embedding tensor. Only the 9
    pinned rows of that tensor are used; it is memory-mapped so the full 1.13 GB is never
    materialised.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import torchvision.transforms as T

    _, primary = concept_map()
    idx = [c["concept_idx"] for c in primary]

    ckpt = hf_hub_download(HF_REPO, "best_model.pt", revision=HF_REV)
    emb_path = hf_hub_download(HF_REPO, "concept_embeddings_368294.pt", revision=HF_REV)
    log("  checkpoint %s" % ckpt)

    try:
        C = torch.load(emb_path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:                       # older torch without mmap
        C = torch.load(emb_path, map_location="cpu")
    if C.shape[0] != 368294:
        raise AssertionError("concept embeddings have %d rows, expected 368294"
                             % C.shape[0])
    Csel = C[idx].float().clone()
    Csel = Csel / Csel.norm(dim=1, keepdim=True).clamp_min(1e-12)
    del C
    log("  pinned %d concept embeddings, dim %d" % tuple(Csel.shape))

    sd = torch.load(ckpt, map_location="cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg",
                           pretrained=False)
    state = sd.get("visual", sd.get("model", sd))
    missing, unexpected = model.load_state_dict(
        {k.replace("visual.", ""): v for k, v in state.items()
         if k.replace("visual.", "") in model.state_dict()}, strict=False)
    log("  loaded image encoder (missing=%d unexpected=%d)"
        % (len(missing), len(unexpected)))
    model = model.to(device).eval()
    Csel = Csel.to(device)

    tf = T.Compose([T.Resize((CLEAR_INPUT_PX, CLEAR_INPUT_PX)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out, t0, bs = [], time.perf_counter(), batch
    k = 0
    while k < len(pids):
        sub = pids[k:k + bs]
        try:
            x = torch.stack([tf(Image.open(IMG_DIR / ("%s.png" % p)).convert("RGB"))
                             for p in sub]).to(device)
            with torch.no_grad():
                f = model(x).float()
                f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
                out.append((f @ Csel.T).cpu().numpy())
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                log("  OOM: batch -> %d" % bs)
                continue
            raise
        k += len(sub)
        if k % 256 == 0:
            log("  %d/%d (%.1f min)" % (k, len(pids), (time.perf_counter() - t0) / 60))
    S = np.concatenate(out)
    if S.shape != (len(pids), len(idx)):
        raise AssertionError("similarities have shape %s, expected (%d, %d)"
                             % (S.shape, len(pids), len(idx)))
    if not np.isfinite(S).all():
        raise AssertionError("non-finite concept similarities")
    return S


# --------------------------------------------------------------------- analysis
def analyse(d, S, names, log=print):
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(SEED)
    y, past, s = d["y"], d["past_tb"], d["clf_score"]
    neg = y == 0
    rows, pvals = [], []

    def boot_rho(a, b):
        v = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(a), len(a))
            if len(np.unique(a[k])) < 2:
                continue
            v.append(spearmanr(a[k], b[k]).statistic)
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    for j, name in enumerate(names):
        c = S[:, j]
        for label, m in (("negatives (all)", neg),
                         ("negatives, past TB", neg & (past == 1)),
                         ("negatives, no past TB", neg & (past == 0)),
                         ("positives (all)", y == 1)):
            if m.sum() < 20:
                continue
            r = spearmanr(c[m], s[m])
            lo, hi = boot_rho(c[m], s[m])
            rows.append({"concept": name, "group": label, "n": int(m.sum()),
                         "spearman_rho": float(r.statistic), "ci95_lo": lo, "ci95_hi": hi,
                         "p_raw": float(r.pvalue)})
            if label == "negatives (all)":
                pvals.append(float(r.pvalue))

    # Benjamini-Hochberg across the concepts, on the primary 'negatives (all)' family
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    padj = np.empty(len(p))
    padj[order] = np.clip(adj, 0, 1)
    it = iter(padj)
    for r in rows:
        r["p_bh_adjusted"] = float(next(it)) if r["group"] == "negatives (all)" else None

    # incremental information over simple clinical/site variables, negatives only
    X0 = np.column_stack([d["rad_system"][neg] == v for v in (1, 2, 3)][:2]
                         + [d["age_group"][neg], d["sex"][neg], d["symptoms"][neg],
                            past[neg]]).astype(float)
    base = LinearRegression().fit(X0, s[neg])
    r2_0 = base.score(X0, s[neg])
    incr = []
    for j, name in enumerate(names):
        X1 = np.column_stack([X0, S[neg, j]])
        r2_1 = LinearRegression().fit(X1, s[neg]).score(X1, s[neg])
        incr.append({"concept": name, "r2_clinical_only": round(r2_0, 4),
                     "r2_with_concept": round(r2_1, 4),
                     "delta_r2": round(r2_1 - r2_0, 4)})
    return rows, incr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-inference", action="store_true",
                    help="actually encode images; omit for a dry validation")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    d = load_tune_only()
    m, primary = concept_map()
    names = [c["concept"] for c in primary]
    print("TUNE cohort: %d patients, %d TB-positive, %d with previous TB"
          % (len(d["pid"]), int(d["y"].sum()), int(d["past_tb"].sum())))
    print("concepts in the primary set: %d of %d (%s excluded, no exact entry)"
          % (len(names), len(m["mapping"]), len(m["mapping"]) - len(names)))
    print("REPORT pids loaded: 0  |  internal_test referenced: no")

    if not a.run_inference:
        print("\nDRY VALIDATION ONLY. Pass --run-inference to encode images.")
        print("Needs ~1.73 GB of downloads and roughly 0.8 GB of host RAM headroom.")
        return

    S = encode_concepts(d["pid"], device=a.device)
    rows, incr = analyse(d, S, names)
    with io.open(RUN / "tables/concept_associations.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with io.open(RUN / "tables/concept_incremental_r2.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(incr[0]))
        w.writeheader()
        w.writerows(incr)
    print("\nwrote aggregate tables. No per-case concept score is written to the bundle.")


if __name__ == "__main__":
    main()
