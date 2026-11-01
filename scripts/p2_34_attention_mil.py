# -*- coding: utf-8 -*-
"""Attention-based MIL and a non-linear scorer: can either beat plain max-pooling?

Max is the crudest aggregator - it takes the single strongest region and ignores the rest.
Two standard upgrades:

  * **gated attention MIL** (Ilse et al. 2018): learn a weight per region and take a
    weighted sum, so several suggestive regions can combine and irrelevant ones are
    suppressed;
  * **a two-layer scorer**, in case the per-region decision is not linearly separable.

**Protocol status.** The primary result (0.7242 on the REPORT half) is already read and
locked. Everything here is selected on the TUNE half and must be reported as **exploratory,
run after the primary result** - it cannot become a new headline, because the held-aside
half has been spent and a second read would no longer be a clean test.

    py -3 scripts/p2_34_attention_mil.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import p2_32_mil_sweep as S                                          # noqa: E402
from p2_cls import selection as SEL                                  # noqa: E402

TB = ("active_tb", "latent_tb", "active_latent_tb")
SEEDS = (17, 29, 43)
OUT = ROOT / "artifacts/p2/regional/attention_mil.json"


class GatedAttentionMIL:
    """Ilse et al. gated attention over regions, with an optional hidden layer."""

    def __init__(self, dim, hidden=128, attn=64, deep=False, seed=0):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.scorer = (torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(),
                                           torch.nn.Linear(hidden, 1)) if deep
                       else torch.nn.Linear(dim, 1)).to(self.dev)
        self.V = torch.nn.Linear(dim, attn).to(self.dev)
        self.U = torch.nn.Linear(dim, attn).to(self.dev)
        self.w = torch.nn.Linear(attn, 1).to(self.dev)

    def params(self):
        return (list(self.scorer.parameters()) + list(self.V.parameters())
                + list(self.U.parameters()) + list(self.w.parameters()))

    def forward(self, x):
        torch = self.torch
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))      # [B, R, 1]
        a = torch.softmax(a, dim=1)
        s = self.scorer(x)                                                # [B, R, 1]
        return (a * s).sum(dim=1).squeeze(-1)

    def train_mode(self, on=True):
        for m in (self.scorer, self.V, self.U, self.w):
            m.train(on)


def train_attn(Xtr, ytr, Xdv, ydv, deep, seed, epochs=120, lr=3e-4, bs=128, wd=1e-4):
    import torch
    m = GatedAttentionMIL(Xtr.shape[2], deep=deep, seed=seed)
    opt = torch.optim.AdamW(m.params(), lr=lr, weight_decay=wd)
    xt = torch.tensor(Xtr, device=m.dev)
    yt = torch.tensor(ytr, dtype=torch.float32, device=m.dev)
    xv = torch.tensor(Xdv, device=m.dev)
    prior = float(yt.mean())
    adj = float(np.log(max(prior, 1e-6) / max(1 - prior, 1e-6)))
    best, best_state, patience = -1.0, None, 0
    for ep in range(epochs):
        m.train_mode(True)
        perm = torch.randperm(len(yt), device=m.dev)
        for i in range(0, len(yt), bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            z = m.forward(xt[k].float())
            torch.nn.functional.binary_cross_entropy_with_logits(z - adj, yt[k]).backward()
            opt.step()
        m.train_mode(False)
        with torch.no_grad():
            sv = torch.cat([m.forward(xv[i:i + 256].float())
                            for i in range(0, len(xv), 256)]).cpu().numpy()
        a = SEL.auc(ydv, sv)
        if a > best + 1e-5:
            best, patience = a, 0
            best_state = [{k2: v.detach().clone() for k2, v in mm.state_dict().items()}
                          for mm in (m.scorer, m.V, m.U, m.w)]
        else:
            patience += 1
            if patience >= 15:
                break
    for mm, st in zip((m.scorer, m.V, m.U, m.w), best_state):
        mm.load_state_dict(st)
    return m, best


def score_attn(m, X, chunk=256):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.tensor(X[i:i + chunk], device=m.dev).float()
            out.append(m.forward(x).cpu().numpy())
    return np.concatenate(out)


def main():
    import pandas as pd
    split = json.load(open(ROOT / "artifacts/p2/external/vietnam_tune_report_split.json",
                           encoding="utf-8"))
    tune = set(split["tune"])
    csv = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")

    with np.load(ROOT / "artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    ti = {i: k for k, i in enumerate(tid)}
    c = c01[c01.image_id.isin(ti)]
    tr = c[(c.split == "train") & (c.category.isin(TB + ("healthy",)))]
    dv = c[c.split == "development"]
    Ttr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, 12)
    Tdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, 12)
    ytr = np.array([1 if x in TB else 0 for x in tr.category])
    ydv = np.array([1 if x in TB else 0 for x in dv.category])
    del TX

    with np.load(ROOT / "artifacts/p2/regional/regional_vietnam_lungs_g12_v1.npz") as z:
        vid, VX = z["image_id"].tolist(), z["reg"]
    sel = [k for k, i in enumerate(vid) if int(i) in tune]
    V = S.repool(VX[sel], 12, 12)
    y = csv.loc[[int(vid[k]) for k in sel], "xpert_result"].to_numpy()
    del VX

    print("TUNE half: %d, %d TB  |  reference: max-pooling 0.7239" % (len(y), y.sum()))
    rows = []
    for name, deep in (("attention", False), ("attention+MLP", True)):
        devs, ps = [], []
        for sd in SEEDS:
            m, d = train_attn(Ttr, ytr, Tdv, ydv, deep, sd)
            devs.append(d)
            ps.append(score_attn(m, V))
        a = float(SEL.auc(y, np.mean(ps, axis=0)))
        rows.append({"method": name, "tbx11k_dev": round(float(np.mean(devs)), 4),
                     "tune_auc": round(a, 4)})
        print("  %-14s TBX11K dev %.4f   TUNE %.4f" % (name, np.mean(devs), a))

    best = max(rows, key=lambda r: r["tune_auc"])
    print("\n  best alternative: %s at %.4f vs max-pooling 0.7239 (%+.4f)"
          % (best["method"], best["tune_auc"], best["tune_auc"] - 0.7239))
    if best["tune_auc"] - 0.7239 < 0.01:
        print("  -> no meaningful improvement; plain max-pooling is already at the ceiling")
        print("     of what this representation supports.")
    else:
        print("  -> a real gain, but EXPLORATORY: the held-aside half is already spent, so")
        print("     this cannot be reported as a new headline without a fresh held-out set.")

    json.dump({"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
               "reference_max_pooling_tune": 0.7239, "results": rows,
               "status": "EXPLORATORY - run after the primary REPORT read"},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
