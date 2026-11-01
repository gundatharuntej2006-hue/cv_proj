# -*- coding: utf-8 -*-
"""Can a different negative class close the screening gap? No new data required.

`p2_vietnam_result.md` localised the failure: on a consecutive screening cohort the model
assigns median 0.886 to Xpert-negatives against 0.953 to cases, i.e. it calls nearly everyone
positive. The reason is visible internally - TB-versus-healthy on TBX11K is AUC 1.0000 while
TB-versus-sick is 0.9861 - and a screening queue is entirely the second case: its negatives
carry healed TB, COPD and chronic change.

If that diagnosis is right, the training negatives are the lever. TBX11K ships 3,800
sick-but-not-tuberculous images, currently diluted by 3,800 healthy ones. This tries four
negative compositions and scores each on all four external sites, with no new data and no
change to the backbone:

  all        healthy + sick, as now (the baseline, Vietnam 0.6151)
  sick_only  drop the healthy negatives entirely
  sick_3x    keep both but weight sick negatives three times
  healthy_only  the opposite extreme, as a control - should be worse everywhere

The prediction, if the diagnosis holds: `sick_only` costs a little on the curated sites
(whose negatives include healthy people) and gains on Vietnam. If instead it changes nothing,
the problem is not the negative distribution and the diagnosis needs revisiting.

Also crosses this with the lungs-only masking, which already transfers better than the full
image at the two most distant sites.

    py -3 scripts/p2_29_hard_negatives.py
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
EXT = ROOT / "artifacts/p2/external"
SHORT = ROOT / "artifacts/p2/shortcut"
EMB = ROOT / "artifacts/p2/emb"
OUT = ROOT / "artifacts/p2/dryrun/hard_negatives.json"
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 2000
TB_CATS = ("active_tb", "latent_tb", "active_latent_tb")

TRAIN_EMB = {"full": EMB / "radino_tbx11k_full_clahe0_c04_v1.npz",
             "lungs_only": SHORT / "radino_tbx11k_lungs_only_v1.npz"}

# external sites: (embedding file, label function)
def nlm_labels(ids):
    return np.array([int(str(i).rsplit("_", 1)[1]) for i in ids])


def vietnam_labels(ids):
    import pandas as pd
    csv = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv")
    csv = csv.set_index("pid")
    return csv.loc[[int(i) for i in ids], "xpert_result"].to_numpy()


SITES = {
    "Shenzhen":   (EXT / "radino_nlm_512_v1.npz", nlm_labels, "shz"),
    "Montgomery": (EXT / "radino_nlm_512_v1.npz", nlm_labels, "mcu"),
    "Vietnam":    (EXT / "radino_vietnam_full_v1.npz", vietnam_labels, None),
}
SITES_LUNGS = {
    "Shenzhen":   (SHORT / "radino_nlm_lungs_only_v1.npz", nlm_labels, "shz"),
    "Montgomery": (SHORT / "radino_nlm_lungs_only_v1.npz", nlm_labels, "mcu"),
    "Vietnam":    (EXT / "radino_vietnam_lungs_only_v1.npz", vietnam_labels, None),
}

COMPOSITIONS = {
    "all": {"healthy": 1.0, "sick_non_tb": 1.0},
    "sick_only": {"sick_non_tb": 1.0},
    "sick_3x": {"healthy": 1.0, "sick_non_tb": 3.0},
    "healthy_only": {"healthy": 1.0},
}


def boot_auc(y, p, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    v = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        v.append(SEL.auc(y[k], p[k]))
    return (float(SEL.auc(y, p)),
            float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def train(variant, comp, log=print):
    """Head trained with the given negative composition. Weighting is by replication,
    which keeps the existing logit-adjusted loss semantics unchanged."""
    import pandas as pd
    with np.load(TRAIN_EMB[variant]) as z:
        ids, X = z["image_id"].tolist(), z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    tr = c01[c01.split == "train"]
    keep = tr[tr.category.isin(TB_CATS)]
    rows = [keep]
    for cat, w in comp.items():
        g = tr[tr.category == cat]
        n = int(round(w))
        rows += [g] * n
        if w != n:                       # fractional weights are not used, but be explicit
            raise ValueError("integer weights only")
    tr2 = pd.concat(rows)
    Xtr = X[[idx[i] for i in tr2.image_id]]
    ytr = H.binary_labels(tr2.category.tolist())
    # development is left untouched so early stopping is comparable across compositions
    dv = c01[c01.split == "development"]
    Xdv, ydv = X[[idx[i] for i in dv.image_id]], H.binary_labels(dv.category.tolist())
    res, auc, _ = H.train_seeds(
        Xtr, ytr, Xdv, ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    return (res[0].state_dict["weight"].numpy().astype(np.float32),
            res[0].state_dict["bias"].numpy().astype(np.float32),
            float(auc), int(len(ytr)), int(ytr.sum()))


def score_sites(W, b, site_map):
    out = {}
    cache = {}
    for name, (path, labfn, prefix) in site_map.items():
        if not path.exists():
            continue
        if path not in cache:
            with np.load(path) as z:
                cache[path] = (z["image_id"].tolist(), z["emb"].astype(np.float32))
        ids, X = cache[path]
        if prefix:
            sel = [k for k, i in enumerate(ids) if str(i).startswith(prefix)]
            ids2 = [ids[k] for k in sel]
            X2 = X[sel]
        else:
            ids2, X2 = ids, X
        y = labfn(ids2)
        p = R.p_tb_from_logits(R.head_logits(X2, W, b), 2)
        out[name] = boot_auc(y, p)
    return out


def main():
    rows = []
    for variant in ("full", "lungs_only"):
        if not TRAIN_EMB[variant].exists():
            continue
        site_map = SITES if variant == "full" else SITES_LUNGS
        print("\n=== %s ===" % variant)
        print("  %-14s %7s %9s  %s" % ("negatives", "n_train", "TBX11K", "external AUC"))
        for comp_name, comp in COMPOSITIONS.items():
            W, b, dev_auc, n_tr, n_pos = train(variant, comp)
            sc = score_sites(W, b, site_map)
            rows.append({"variant": variant, "negatives": comp_name, "n_train": n_tr,
                         "tbx11k_dev_auc": round(dev_auc, 4),
                         "external": {k: [round(v[0], 4), round(v[1], 4), round(v[2], 4)]
                                      for k, v in sc.items()}})
            ext = "  ".join("%s %.4f" % (k[:4], v[0]) for k, v in sc.items())
            print("  %-14s %7d %9.4f  %s" % (comp_name, n_tr, dev_auc, ext))

    # the comparison that decides it
    print("\n" + "=" * 66)
    print("Vietnam (the screening cohort) by negative composition:")
    print("  %-12s %-14s %9s %-20s" % ("variant", "negatives", "AUC", "95% CI"))
    best = None
    for r in rows:
        v = r["external"].get("Vietnam")
        if not v:
            continue
        print("  %-12s %-14s %9.4f [%.4f, %.4f]"
              % (r["variant"], r["negatives"], v[0], v[1], v[2]))
        if best is None or v[0] > best[0]:
            best = (v[0], r["variant"], r["negatives"])
    base = next((r["external"]["Vietnam"][0] for r in rows
                 if r["variant"] == "full" and r["negatives"] == "all"), None)
    print("\ninterpretation:")
    if best and base:
        gain = best[0] - base
        print("  baseline (full / all negatives): %.4f" % base)
        print("  best: %s / %s at %.4f  (%+.4f)" % (best[1], best[2], best[0], gain))
        if gain >= 0.04:
            print("  -> the negative composition IS the lever; the diagnosis holds and this")
            print("     is a real partial fix requiring no new data.")
        elif gain >= 0.015:
            print("  -> a modest but real gain; worth reporting, not a solution.")
        else:
            print("  -> changing the negatives does not help. The diagnosis was incomplete:")
            print("     the deficit is in the representation or the positives, not the")
            print("     negative distribution.")
        print("  commercial comparator on the same cohort: 0.7827")

    json.dump({"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
               "compositions": {k: v for k, v in COMPOSITIONS.items()},
               "results": rows,
               "note": ("Weighting by replication. Development is held fixed across "
                        "compositions so early stopping stays comparable. EXPLORATORY.")},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
