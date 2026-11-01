# -*- coding: utf-8 -*-
"""Would more training data of the same kind raise the screening number?

A direct question with a cheap answer. Train on increasing fractions of TBX11K and watch
what happens at each external site. If external performance has plateaued by 100%, more
data like this buys nothing and the limit is the representation or the head. If it is still
climbing, data is the lever and acquiring more is worth the effort.

Subsampling is stratified and seeded, and repeated over three draws at each fraction so the
curve is not one lucky subset. The best-known configuration is used throughout: lungs-only
masking with healthy-only negatives.

    py -3 scripts/p2_30_learning_curve.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import heads as H                                       # noqa: E402
from p2_cls import registry as R                                    # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

TRAIN = ROOT / "artifacts/p2/shortcut/radino_tbx11k_lungs_only_v1.npz"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT = ROOT / "artifacts/p2/dryrun/learning_curve.json"
TB = ("active_tb", "latent_tb", "active_latent_tb")
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DRAWS = 3
SEEDS = (17, 29, 43)


def main():
    import pandas as pd
    with np.load(TRAIN) as z:
        ids, X = z["image_id"].tolist(), z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    tr = c01[c01.split == "train"]
    dv = c01[c01.split == "development"]
    Xdv, ydv = X[[idx[i] for i in dv.image_id]], H.binary_labels(dv.category.tolist())

    # best-known configuration: healthy-only negatives
    pos = tr[tr.category.isin(TB)]
    neg = tr[tr.category == "healthy"]

    # external sites
    csv = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv").set_index("pid")
    sites = {}
    p = ROOT / "artifacts/p2/external/radino_vietnam_lungs_only_v1.npz"
    if p.exists():
        with np.load(p) as z:
            pid = [int(i) for i in z["image_id"].tolist()]
            sites["Vietnam"] = (z["emb"].astype(np.float32),
                                csv.loc[pid, "xpert_result"].to_numpy())
    p = ROOT / "artifacts/p2/shortcut/radino_nlm_lungs_only_v1.npz"
    if p.exists():
        with np.load(p) as z:
            nid, NX = z["image_id"].tolist(), z["emb"].astype(np.float32)
        for pre, nm in (("shz", "Shenzhen"), ("mcu", "Montgomery")):
            sel = [k for k, i in enumerate(nid) if str(i).startswith(pre)]
            sites[nm] = (NX[sel], np.array([int(str(nid[k]).rsplit("_", 1)[1]) for k in sel]))

    print("learning curve: lungs-only masking, healthy-only negatives, %d draws per point"
          % DRAWS)
    print("  %-6s %7s %9s  %s" % ("frac", "n_train", "TBX11K", "  ".join("%-16s" % s for s in sites)))
    rows = []
    for frac in FRACTIONS:
        acc = {s: [] for s in sites}
        dev_accs = []
        for draw in range(DRAWS if frac < 1.0 else 1):
            rng = np.random.default_rng(1000 + draw)
            kp = rng.choice(len(pos), max(int(round(len(pos) * frac)), 2), replace=False)
            kn = rng.choice(len(neg), max(int(round(len(neg) * frac)), 2), replace=False)
            sub = pd.concat([pos.iloc[kp], neg.iloc[kn]])
            Xtr = X[[idx[i] for i in sub.image_id]]
            ytr = H.binary_labels(sub.category.tolist())
            res, dauc, _ = H.train_seeds(
                Xtr, ytr, Xdv, ydv,
                H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
                seeds=SEEDS)
            W = res[0].state_dict["weight"].numpy().astype(np.float32)
            b = res[0].state_dict["bias"].numpy().astype(np.float32)
            dev_accs.append(dauc)
            for s, (SX, sy) in sites.items():
                acc[s].append(float(SEL.auc(sy, R.p_tb_from_logits(R.head_logits(SX, W, b), 2))))
        n_tr = len(pos) + len(neg)
        row = {"fraction": frac, "n_train": int(round(n_tr * frac)),
               "tbx11k_dev": round(float(np.mean(dev_accs)), 4),
               "external": {s: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
                            for s, v in acc.items()}}
        rows.append(row)
        print("  %-6.2f %7d %9.4f  %s"
              % (frac, row["n_train"], row["tbx11k_dev"],
                 "  ".join("%.4f+-%.3f  " % (row["external"][s][0], row["external"][s][1])
                           for s in sites)))

    print("\ninterpretation:")
    for s in sites:
        a, b_ = rows[-2]["external"][s][0], rows[-1]["external"][s][0]
        first, last = rows[0]["external"][s][0], rows[-1]["external"][s][0]
        slope = b_ - a                      # gain over the final 25% of the data
        print("  %-11s 25%%->100%%: %+.4f | last quarter: %+.4f -> %s"
              % (s, last - first, slope,
                 "still climbing" if slope > 0.01 else
                 "plateaued" if abs(slope) <= 0.01 else "declining"))
    print("\n  A plateau means more data of this kind will not help and the limit is the")
    print("  frozen representation or the linear head. Still climbing means data is the lever.")

    json.dump({"created": time.strftime("%Y-%m-%d"), "config": "lungs_only + healthy_only",
               "draws": DRAWS, "seeds": list(SEEDS), "curve": rows},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
