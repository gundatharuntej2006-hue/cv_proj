# -*- coding: utf-8 -*-
"""Which negatives? The choice of control group decides most of the headline number.

Every AUC reported so far uses "TB versus everything else", where everything else is 3,800
healthy and 3,800 sick-but-not-TB images. Those two control groups pose completely different
problems:

  - **TB vs healthy** is the easy, flattering comparison, and it is the one most published
    TB screening numbers are really measuring.
  - **TB vs sick_non_tb** is the clinically honest one. A screening queue is full of people
    with *something* wrong with their chest. Telling TB from pneumonia, fibrosis or a tumour
    is the actual task; telling TB from a normal film is not.

This also localises the border shortcut without downloading anything. All three groups come
from the same hospitals, the same scanners and the same pipeline, so if the border signal
survives the switch from healthy to sick controls, it is not simply "TB images were acquired
differently from normal images".

Two complementary measurements:

  (a) the **deployed** head, trained once on TB versus all negatives, with its scores then
      restricted to each control group - what the system we actually built does;
  (b) a **dedicated** head trained on each contrast - what is learnable at all.

    py -3 scripts/p2_25_control_group.py
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

SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT = ROOT / "artifacts/p2/dryrun/control_group.json"
SHORT = ROOT / "artifacts/p2/shortcut"
VARIANTS = {
    "full":        ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz",
    "lungs_only":  SHORT / "radino_tbx11k_lungs_only_v1.npz",
    "border_only": SHORT / "radino_tbx11k_border_only_v1.npz",
}
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 2000
TB_CATS = ("active_tb", "latent_tb", "active_latent_tb")


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


def load(path):
    import pandas as pd
    with np.load(path) as z:
        ids, X = z["image_id"].tolist(), z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)].copy()
    c01["row"] = [idx[i] for i in c01.image_id]
    return X, c01


def train(X, c01, train_cats, log=print):
    """Head trained on the train split restricted to TB plus the given negative categories."""
    tr = c01[(c01.split == "train") & (c01.category.isin(TB_CATS + train_cats))]
    dv = c01[(c01.split == "development") & (c01.category.isin(TB_CATS + train_cats))]
    Xtr, ytr = X[tr.row.to_numpy()], H.binary_labels(tr.category.tolist())
    Xdv, ydv = X[dv.row.to_numpy()], H.binary_labels(dv.category.tolist())
    res, auc, sd = H.train_seeds(
        Xtr, ytr, Xdv, ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    return (res[0].state_dict["weight"].numpy().astype(np.float32),
            res[0].state_dict["bias"].numpy().astype(np.float32), float(auc))


def main():
    contrasts = {
        "TB vs all negatives": ("healthy", "sick_non_tb"),
        "TB vs healthy only": ("healthy",),
        "TB vs sick_non_tb only": ("sick_non_tb",),
    }
    rows = []

    for vname, path in VARIANTS.items():
        if not path.exists():
            print("skipping %s (not embedded)" % vname)
            continue
        X, c01 = load(path)
        print("\n=== %s ===" % vname)

        # (a) the deployed head: trained once on TB vs all, scored on each control group
        W, b, dev_auc = train(X, c01, ("healthy", "sick_non_tb"))
        print("  deployed head (trained on TB vs all), scored against each control group:")
        for label, cats in contrasts.items():
            g = c01[(c01.split == "development") & (c01.category.isin(TB_CATS + cats))]
            y = H.binary_labels(g.category.tolist())
            p = R.p_tb_from_logits(R.head_logits(X[g.row.to_numpy()], W, b), 2)
            a, lo, hi = boot_auc(y, p)
            rows.append({"variant": vname, "mode": "deployed", "contrast": label,
                         "n": int(len(y)), "n_tb": int(y.sum()),
                         "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)]})
            print("    %-24s n=%4d  AUC %.4f [%.4f, %.4f]" % (label, len(y), a, lo, hi))

        # (b) dedicated heads: what each contrast can support when trained for it
        print("  dedicated heads (trained on that contrast):")
        for label, cats in contrasts.items():
            Wc, bc, _ = train(X, c01, cats)
            g = c01[(c01.split == "development") & (c01.category.isin(TB_CATS + cats))]
            y = H.binary_labels(g.category.tolist())
            p = R.p_tb_from_logits(R.head_logits(X[g.row.to_numpy()], Wc, bc), 2)
            a, lo, hi = boot_auc(y, p)
            rows.append({"variant": vname, "mode": "dedicated", "contrast": label,
                         "n": int(len(y)), "n_tb": int(y.sum()),
                         "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)]})
            print("    %-24s n=%4d  AUC %.4f [%.4f, %.4f]" % (label, len(y), a, lo, hi))

    # ---------------------------------------------------------------- interpretation
    def get(v, mode, label):
        for r in rows:
            if r["variant"] == v and r["mode"] == mode and r["contrast"] == label:
                return r
        return None

    print("\n" + "=" * 72)
    print("summary (dedicated heads, the fairest comparison):")
    print("  %-12s %14s %14s %12s" % ("variant", "vs healthy", "vs sick_non_tb", "drop"))
    summary = {}
    for v in VARIANTS:
        hh, ss = get(v, "dedicated", "TB vs healthy only"), \
            get(v, "dedicated", "TB vs sick_non_tb only")
        if not hh or not ss:
            continue
        drop = hh["auc"] - ss["auc"]
        summary[v] = {"vs_healthy": hh["auc"], "vs_sick": ss["auc"], "drop": round(drop, 4)}
        print("  %-12s %14.4f %14.4f %12.4f" % (v, hh["auc"], ss["auc"], drop))

    print("\ninterpretation:")
    if "full" in summary:
        d = summary["full"]["drop"]
        print("  The full model loses %.4f AUC when the controls are sick rather than "
              "healthy." % d)
        if d >= 0.05:
            print("  That is a large share of the headline number: much of the reported "
                  "performance is\n  'TB looks different from a normal chest', not 'TB "
                  "looks different from other disease'.")
        else:
            print("  The task is nearly as hard against sick controls, which is a genuinely "
                  "good sign.")
    if "border_only" in summary:
        b = summary["border_only"]
        print("  The border model scores %.4f against healthy and %.4f against sick "
              "controls." % (b["vs_healthy"], b["vs_sick"]))
        if b["vs_sick"] >= 0.75:
            print("  It stays high against sick controls drawn from the same hospitals, so "
                  "the border signal\n  is not merely 'TB films were acquired unlike normal "
                  "films'.")
        else:
            print("  It collapses against sick controls, which localises the shortcut: the "
                  "border separates\n  the TB collection group from the healthy collection "
                  "group specifically.")

    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "results": rows, "summary": summary,
           "note": ("All three groups share hospitals, scanners and pipeline, so the "
                    "healthy/sick swap changes only the control population. Provisional "
                    "split, development only; internal_test untouched.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
