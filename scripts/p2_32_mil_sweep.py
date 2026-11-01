# -*- coding: utf-8 -*-
"""Sweep the focal-pooling design, selecting on a held-aside half of the screening cohort.

`p2_focal_pooling.md` showed max-over-regions beats global pooling by +0.031 on Vietnam, with
an untuned 6x6 grid and a deliberately simple trainer that underfits by about 0.08 AUC on
development. This searches the design properly.

**The methodological problem, and the safeguard.** Tuning against Vietnam would overfit a
test set - precisely the failure this project exists to document. So the cohort was split
once, before any tuning, stratified by outcome and scanner and seeded:

    TUNE   1,352 participants, 139 TB   <- every decision below is made on this half
    REPORT 1,356 participants, 141 TB   <- read once, at the end, for the final number

The split is frozen in `artifacts/p2/external/vietnam_tune_report_split.json`. Nothing in
this script reads the REPORT half; `p2_33_mil_final.py` does that exactly once.

Searched: region grid (derived from a single 12x12 embedding by further pooling), aggregator,
and the training schedule. Mini-batch training with early stopping on TBX11K development
replaces the full-batch loop, which is where the underfitting came from.

    py -3 scripts/p2_32_mil_sweep.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import selection as SEL                                 # noqa: E402

REG = ROOT / "artifacts/p2/regional"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
SPLIT_JSON = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
OUT = REG / "mil_sweep.json"
TB = ("active_tb", "latent_tb", "active_latent_tb")
SEEDS = (17, 29, 43)
GRIDS = (1, 2, 3, 4, 6, 9, 12, 18)
AGGS = ("mean", "max", "top3", "top5", "lse")


def repool(X, src, dst, chunk=512):
    """[N, src*src, C] -> [N, dst*dst, C] by average pooling, then re-normalise.

    Chunked and kept in float16. A whole-array float32 cast is 3 GB at the 18-grid and
    exhausts memory on this machine; the trainer casts each mini-batch instead.
    """
    if src % dst:
        raise ValueError("%d does not divide %d" % (dst, src))
    f = src // dst
    n, _, c = X.shape
    out = np.empty((n, dst * dst, c), np.float16)
    for s0 in range(0, n, chunk):
        g = X[s0:s0 + chunk].astype(np.float32)
        m = len(g)
        if f > 1:
            g = g.reshape(m, src, src, c).reshape(m, dst, f, dst, f, c).mean(axis=(2, 4))
            g = g.reshape(m, dst * dst, c)
        g /= np.maximum(np.linalg.norm(g, axis=2, keepdims=True), 1e-9)
        out[s0:s0 + chunk] = g.astype(np.float16)
    return out


def agg_torch(s, how, torch):
    if how == "mean":
        return s.mean(1)
    if how == "max":
        return s.max(1).values
    if how == "top3":
        return s.topk(min(3, s.shape[1]), dim=1).values.mean(1)
    if how == "top5":
        return s.topk(min(5, s.shape[1]), dim=1).values.mean(1)
    if how == "lse":
        return torch.logsumexp(s, dim=1) - float(np.log(s.shape[1]))
    raise ValueError(how)


def agg_np(s, how):
    if how == "mean":
        return s.mean(1)
    if how == "max":
        return s.max(1)
    if how == "top3":
        return np.sort(s, 1)[:, -min(3, s.shape[1]):].mean(1)
    if how == "top5":
        return np.sort(s, 1)[:, -min(5, s.shape[1]):].mean(1)
    if how == "lse":
        m = s.max(1, keepdims=True)
        return (m + np.log(np.exp(s - m).sum(1, keepdims=True)))[:, 0] - np.log(s.shape[1])
    raise ValueError(how)


def train(Xtr, ytr, Xdv, ydv, how, seed, epochs=120, lr=3e-3, bs=256, wd=1e-4):
    """Mini-batch Adam with early stopping on development. Replaces the full-batch loop."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Xtr, device=dev)                 # float16, cast per batch
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    xv = torch.tensor(Xdv, device=dev)
    lin = torch.nn.Linear(xt.shape[2], 1).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    prior = float(yt.mean())
    adj = float(np.log(max(prior, 1e-6) / max(1 - prior, 1e-6)))
    n = len(yt)
    best, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        lin.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            z = agg_torch(lin(xt[k].float()).squeeze(-1), how, torch)
            torch.nn.functional.binary_cross_entropy_with_logits(z - adj, yt[k]).backward()
            opt.step()
        lin.eval()
        with torch.no_grad():
            sv = torch.cat([lin(xv[i:i+512].float()).squeeze(-1)
                            for i in range(0, len(xv), 512)]).cpu().numpy()
        a = SEL.auc(ydv, agg_np(sv, how))
        if a > best + 1e-5:
            best, patience = a, 0
            best_state = {k2: v.detach().clone() for k2, v in lin.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break
    lin.load_state_dict(best_state)
    return (lin.weight.detach().cpu().numpy(), lin.bias.detach().cpu().numpy(), best)


def score(X, W, b, how, chunk=512):
    out = []
    for i in range(0, len(X), chunk):
        out.append(agg_np((X[i:i + chunk].astype(np.float32) @ W.T + b)[:, :, 0], how))
    return np.concatenate(out)


def main():
    import pandas as pd
    def load(src):
        a = REG / ("regional_tbx11k_lungs_g%d_v1.npz" % src)
        b = REG / ("regional_vietnam_lungs_g%d_v1.npz" % src)
        if not a.exists() or not b.exists():
            return None
        with np.load(a) as z:
            t = (z["image_id"].tolist(), z["reg"])
        with np.load(b) as z:
            v = (z["image_id"].tolist(), z["reg"])
        return t, v

    base = load(12)
    fine = load(18)                      # 18 gives grids 9 and 18, not poolable from 12
    if base is None:
        raise SystemExit("run p2_31 --stage embed at GRID=12 first")
    (tid, TX), (vid, VX) = base

    split = json.load(open(SPLIT_JSON, encoding="utf-8"))
    tune = set(split["tune"])
    csv = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
    vsel = [k for k, i in enumerate(vid) if int(i) in tune]
    y_tune = csv.loc[[int(vid[k]) for k in vsel], "xpert_result"].to_numpy()
    print("TUNE half: %d participants, %d TB  (REPORT half untouched)"
          % (len(vsel), y_tune.sum()))

    ti = {i: k for k, i in enumerate(tid)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(ti)]
    tr = c01[(c01.split == "train") & (c01.category.isin(TB + ("healthy",)))]
    dv = c01[c01.split == "development"]
    itr = [ti[i] for i in tr.image_id]
    idv = [ti[i] for i in dv.image_id]
    ytr = np.array([1 if c in TB else 0 for c in tr.category], np.int64)
    ydv = np.array([1 if c in TB else 0 for c in dv.category], np.int64)

    rows = []
    print("\n  %-5s %-6s %9s %9s" % ("grid", "agg", "TBX11Kdev", "TUNE"))
    t0 = time.perf_counter()
    for g in GRIDS:
        if g in (9, 18):
            # 9 and 18 divide 18 but not 12, so they come from the finer embedding
            if fine is None:
                print("  (18-grid embedding not present; grid %d skipped)" % g)
                continue
            (tid2, TX2), (vid2, VX2) = fine
            ti2 = {i: k for k, i in enumerate(tid2)}
            Ttr = repool(TX2[[ti2[i] for i in tr.image_id]], 18, g)
            Tdv = repool(TX2[[ti2[i] for i in dv.image_id]], 18, g)
            vsel2 = [k for k, i in enumerate(vid2) if int(i) in tune]
            Vt = repool(VX2[vsel2], 18, g)
        else:
            Ttr = repool(TX[itr], 12, g)
            Tdv = repool(TX[idv], 12, g)
            Vt = repool(VX[vsel], 12, g)
        for how in AGGS:
            if g == 1 and how != "mean":
                continue                      # one region: all aggregators coincide
            devs, ps = [], []
            for sd in SEEDS:
                W, b, d = train(Ttr, ytr, Tdv, ydv, how, sd)
                devs.append(d)
                ps.append(score(Vt, W, b, how))
            a = float(SEL.auc(y_tune, np.mean(ps, axis=0)))
            rows.append({"grid": g, "agg": how, "regions": g * g,
                         "tbx11k_dev": round(float(np.mean(devs)), 4),
                         "tune_auc": round(a, 4)})
            print("  %-5d %-6s %9.4f %9.4f" % (g, how, np.mean(devs), a))

    rows.sort(key=lambda r: -r["tune_auc"])
    best = rows[0]
    print("\n(%.1f min)" % ((time.perf_counter() - t0) / 60))
    print("best on TUNE: grid %d, %s -> %.4f (TBX11K dev %.4f)"
          % (best["grid"], best["agg"], best["tune_auc"], best["tbx11k_dev"]))
    print("runner-up:    grid %d, %s -> %.4f"
          % (rows[1]["grid"], rows[1]["agg"], rows[1]["tune_auc"]))
    print("\nglobal-pooling control (grid 1): %.4f"
          % next(r["tune_auc"] for r in rows if r["grid"] == 1))
    print("\nThis configuration is now FROZEN. p2_33 scores it once on the REPORT half.")

    json.dump({"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
               "selected_on": "vietnam TUNE half only", "results": rows, "best": best},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
