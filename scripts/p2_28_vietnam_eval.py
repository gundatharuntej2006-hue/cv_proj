# -*- coding: utf-8 -*-
"""Evaluate the frozen TBX11K model on the Vietnam screening cohort.

The fourth external site, and the decisive one for the shortcut claim: the CSV labels the
radiography system and **the scanner does not predict the outcome** (AUC 0.4722, chance), so
a border-only model that beats chance here cannot be reading the machine.

Reports, for each of the three maskings:
  - AUC with a bootstrap interval, against the Lunit INSIGHT commercial comparator scored on
    the same patients;
  - a per-scanner breakdown, which no other site here supports;
  - sensitivity and specificity at the TBX11K-fitted threshold, i.e. what would happen on
    installation with no local recalibration.

Nothing here is trained on. The heads come from TBX11K and are applied unchanged.

    py -3 scripts/p2_28_vietnam_eval.py
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

CSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
BUNDLE = ROOT / "artifacts/p2/bundle/bundle.json"
EXT = ROOT / "artifacts/p2/external"
SHORT = ROOT / "artifacts/p2/shortcut"
OUT = EXT / "vietnam_eval.json"
TBX = {"full": ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz",
       "lungs_only": SHORT / "radino_tbx11k_lungs_only_v1.npz",
       "border_only": SHORT / "radino_tbx11k_border_only_v1.npz"}
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 2000
SCANNER = {1: "JPI Healthcare", 2: "DRTECH", 3: "Konica Minolta"}


def boot_auc(y, p, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    vals = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        vals.append(SEL.auc(y[k], p[k]))
    return (float(SEL.auc(y, p)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def train_head(path):
    import pandas as pd
    with np.load(path) as z:
        ids, X = z["image_id"].tolist(), z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    tr, dv = c01[c01.split == "train"], c01[c01.split == "development"]
    res, auc, _ = H.train_seeds(
        X[[idx[i] for i in tr.image_id]], H.binary_labels(tr.category.tolist()),
        X[[idx[i] for i in dv.image_id]], H.binary_labels(dv.category.tolist()),
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    return (res[0].state_dict["weight"].numpy().astype(np.float32),
            res[0].state_dict["bias"].numpy().astype(np.float32), float(auc))


def main():
    import pandas as pd
    csv = pd.read_csv(CSV).set_index("pid")
    meta = json.load(open(BUNDLE, encoding="utf-8"))
    T = meta.get("temperature") or 1.0
    t_lower = meta.get("t_lower")

    print("Vietnam cohort: %d participants, %d Xpert-positive (%.1f%%)"
          % (len(csv), csv.xpert_result.sum(), 100 * csv.xpert_result.mean()))

    # the commercial comparator, on exactly the patients we score
    rows, scores = [], {}
    for variant in ("full", "lungs_only", "border_only"):
        p_emb = EXT / ("radino_vietnam_%s_v1.npz" % variant)
        if not p_emb.exists():
            continue
        with np.load(p_emb) as z:
            pids = [int(i) for i in z["image_id"].tolist()]
            X = z["emb"].astype(np.float32)
        W, b, dev_auc = train_head(TBX[variant])
        p = R.p_tb_from_logits(R.head_logits(X, W, b), 2)
        sub = csv.loc[pids]
        y = sub.xpert_result.to_numpy()
        scores[variant] = (pids, p, y, sub)
        a, lo, hi = boot_auc(y, p)
        rows.append({"variant": variant, "scope": "all", "n": len(y), "n_tb": int(y.sum()),
                     "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
                     "tbx11k_dev_auc": round(dev_auc, 4)})
        print("  %-12s n=%4d TB=%3d  AUC %.4f [%.4f, %.4f]"
              % (variant, len(y), y.sum(), a, lo, hi))

    pids, p_full, y, sub = scores["full"]

    print("\ncommercial comparator on the same patients:")
    for c, label in (("lunit_3100", "Lunit 3.1.0.0"), ("lunit_3901", "Lunit 3.9.0.1")):
        a, lo, hi = boot_auc(y, sub[c].to_numpy())
        rows.append({"variant": c, "scope": "all", "n": len(y), "n_tb": int(y.sum()),
                     "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)]})
        print("  %-14s AUC %.4f [%.4f, %.4f]" % (label, a, lo, hi))

    print("\nper scanner (full-image model):")
    for s in sorted(sub.rad_system.unique()):
        m = (sub.rad_system == s).to_numpy()
        if len(set(y[m].tolist())) < 2:
            continue
        a, lo, hi = boot_auc(y[m], p_full[m])
        al, _, _ = boot_auc(y[m], sub.lunit_3901.to_numpy()[m])
        rows.append({"variant": "full", "scope": "scanner_%d" % s, "n": int(m.sum()),
                     "n_tb": int(y[m].sum()), "auc": round(a, 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "lunit_3901": round(al, 4)})
        print("  %-16s n=%4d TB=%3d  ours %.4f [%.4f, %.4f] | Lunit %.4f"
              % (SCANNER.get(s, s), m.sum(), y[m].sum(), a, lo, hi, al))

    # what installation without recalibration would look like
    def cal(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-np.log(x / (1 - x)) / T))
    pc = cal(p_full)
    sens = float((pc[y == 1] >= t_lower).mean())
    spec = float((pc[y == 0] < t_lower).mean())
    s70 = SEL.sensitivity_at_specificity(y, p_full, 0.70)[0]
    s95 = SEL.sensitivity_at_specificity(y, p_full, 0.95)[0]
    print("\nat the TBX11K threshold, no local recalibration: sens %.4f | spec %.4f"
          % (sens, spec))
    print("with local recalibration: sens@70%%spec %.4f | sens@95%%spec %.4f" % (s70, s95))

    print("\nthe four-site picture (full / lungs-only / border-only):")
    print("  %-18s %8s %12s %12s" % ("site", "full", "lungs-only", "border-only"))
    for name, f, l, bo in (("TBX11K (home)", 0.9949, 0.9889, 0.9094),
                           ("Shenzhen, China", 0.9382, 0.9284, 0.5209),
                           ("Montgomery, USA", 0.8183, 0.8554, 0.6082),
                           ("DA+DB, India", 0.7932, 0.7791, 0.6240)):
        print("  %-18s %8.4f %12.4f %12.4f" % (name, f, l, bo))
    g = {r["variant"]: r["auc"] for r in rows if r["scope"] == "all"}
    print("  %-18s %8.4f %12.4f %12.4f"
          % ("Vietnam", g.get("full", 0), g.get("lungs_only", 0), g.get("border_only", 0)))

    bo = g.get("border_only")
    print("\ninterpretation:")
    print("  Scanner does not predict TB here (AUC 0.4722), so a border signal cannot be")
    print("  the machine. Border-only scores %.4f against its TBX11K home value of 0.9094." % bo)
    if bo < 0.60:
        print("  -> the shortcut is absent on a cohort where acquisition cannot encode the label.")
    elif bo < 0.70:
        print("  -> a weak residual remains; it is not the scanner, so it is a patient-level cue.")
    else:
        print("  -> a substantial border signal survives; this needs explaining.")

    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "cohort": {"n": int(len(csv)), "n_tb": int(csv.xpert_result.sum())},
           "results": rows,
           "operating_point": {"t_lower": t_lower, "sensitivity": round(sens, 4),
                               "specificity": round(spec, 4),
                               "sens_at_70_spec_local": round(float(s70), 4),
                               "sens_at_95_spec_local": round(float(s95), 4)},
           "note": ("EXPLORATORY. Heads trained on TBX11K and applied unchanged. Xpert "
                    "Ultra reference standard. internal_test untouched.")}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
