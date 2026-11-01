# -*- coding: utf-8 -*-
"""Sensitivity by TB subtype. Reported nowhere until now, and it changes what the number means.

`heads.binary_labels` maps **every** TB subtype to positive: `active_tb`, `latent_tb` and
`active_latent_tb` are all label 1. So the headline sensitivity is a weighted blend of two
clinically different findings:

  - **active TB** — transmissible, needs treatment now. This is what a screening programme
    is for.
  - **latent TB** — healed or old lesions, often calcified and radiographically distinctive.
    Not infectious, not urgent.

If latent TB is the *easier* of the two to spot — which is plausible, since calcified scars
are high-contrast and obvious — then the blended sensitivity **overstates** the system's
ability to find the cases that matter. The report currently quotes the blend.

This measures both, with Wilson intervals, at two operating points: the bundle's fitted
threshold and the threshold giving 95% specificity on the same split.

    py -3 scripts/p2_20_tb_subgroups.py

Development and calibration are reported separately rather than pooled: the thresholds were
fitted on calibration, so pooling would mix an in-sample and an out-of-sample estimate.
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

EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
BUNDLE = ROOT / "artifacts/p2/bundle/bundle.json"
OUT = ROOT / "artifacts/p2/dryrun/tb_subgroups.json"
SEEDS = (17, 29, 43, 61, 79)
SUBTYPES = ("active_tb", "latent_tb", "active_latent_tb")
SPEC = 0.95


def main():
    import pandas as pd
    with np.load(EMB) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    meta = json.load(open(BUNDLE, encoding="utf-8"))
    T = meta.get("temperature") or 1.0
    t_lower = meta.get("t_lower")

    # train exactly as every other run does: train split, development for early stopping
    tr = c01[c01.split == "train"]
    Xtr = emb[[idx[i] for i in tr.image_id]]
    ytr = H.binary_labels(tr.category.tolist())
    dv = c01[c01.split == "development"]
    Xdv = emb[[idx[i] for i in dv.image_id]]
    ydv = H.binary_labels(dv.category.tolist())
    res, mean_auc, sd = H.train_seeds(
        Xtr, ytr, Xdv, ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    W = res[0].state_dict["weight"].numpy().astype(np.float32)
    b = res[0].state_dict["bias"].numpy().astype(np.float32)
    print("head: dev AUC %.4f (seed sd %.4f)" % (mean_auc, sd))

    def calibrated(X):
        p = R.p_tb_from_logits(R.head_logits(X, W, b), 2)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        lg = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-lg / T))

    rows = []
    for split in ("development", "calibration"):
        g = c01[c01.split == split]
        if g.empty:
            continue
        X = emb[[idx[i] for i in g.image_id]]
        p = calibrated(X)
        y = H.binary_labels(g.category.tolist())
        cat = g.category.to_numpy()
        # the threshold giving 95% specificity on this split's own negatives
        thr95 = float(np.quantile(p[y == 0], SPEC))
        print("\n%s (n=%d): t_lower %.4f | thr@95%%spec %.4f"
              % (split, len(g), t_lower, thr95))
        print("  %-18s %5s %10s %22s %22s"
              % ("subtype", "n", "median p", "sens @ t_lower", "sens @ 95% spec"))
        for st in SUBTYPES:
            m = cat == st
            n = int(m.sum())
            if n == 0:
                continue
            k_lo = int((p[m] >= t_lower).sum())
            k_95 = int((p[m] >= thr95).sum())
            a, bb = SEL.wilson_ci(k_lo, n)
            c, d = SEL.wilson_ci(k_95, n)
            rows.append({"split": split, "subtype": st, "n": n,
                         "median_p": round(float(np.median(p[m])), 4),
                         "sens_at_t_lower": round(k_lo / n, 4),
                         "ci_at_t_lower": [round(a, 4), round(bb, 4)],
                         "sens_at_95_spec": round(k_95 / n, 4),
                         "ci_at_95_spec": [round(c, 4), round(d, 4)],
                         "threshold_95": round(thr95, 6)})
            print("  %-18s %5d %10.4f   %.4f [%.4f, %.4f]   %.4f [%.4f, %.4f]"
                  % (st, n, np.median(p[m]), k_lo / n, a, bb, k_95 / n, c, d))
        # the blended number the report currently quotes
        mt = y == 1
        k_lo, n_t = int((p[mt] >= t_lower).sum()), int(mt.sum())
        a, bb = SEL.wilson_ci(k_lo, n_t)
        rows.append({"split": split, "subtype": "ALL_TB_blended", "n": n_t,
                     "median_p": round(float(np.median(p[mt])), 4),
                     "sens_at_t_lower": round(k_lo / n_t, 4),
                     "ci_at_t_lower": [round(a, 4), round(bb, 4)],
                     "sens_at_95_spec": round(float((p[mt] >= thr95).mean()), 4),
                     "ci_at_95_spec": None, "threshold_95": round(thr95, 6)})
        print("  %-18s %5d %10.4f   %.4f [%.4f, %.4f]   (what the report quotes)"
              % ("ALL TB (blended)", n_t, np.median(p[mt]), k_lo / n_t, a, bb))

    # the comparison that matters: active versus latent, pooled across both splits for power
    print("\nactive vs latent, pooled over development + calibration (for power):")
    pooled = {}
    for st in ("active_tb", "latent_tb"):
        n = sum(r["n"] for r in rows if r["subtype"] == st)
        k = sum(int(round(r["sens_at_t_lower"] * r["n"])) for r in rows if r["subtype"] == st)
        lo, hi = SEL.wilson_ci(k, n)
        pooled[st] = {"n": n, "detected": k, "sens": round(k / n, 4),
                      "ci95": [round(lo, 4), round(hi, 4)]}
        print("  %-12s %3d/%3d = %.4f [%.4f, %.4f]" % (st, k, n, k / n, lo, hi))
    if len(pooled) == 2:
        a, l = pooled["active_tb"], pooled["latent_tb"]
        overlap = not (a["ci95"][1] < l["ci95"][0] or l["ci95"][1] < a["ci95"][0])
        print("  -> %s" % ("intervals overlap; no difference demonstrated at this sample size"
                           if overlap else
                           "DIFFERENT: the blended sensitivity is not representative"))
        if l["sens"] > a["sens"]:
            print("     latent TB is detected MORE often than active TB, so the blended "
                  "number flatters\n     the clinically important case.")

    # Can the model tell active from latent at all? It was never trained to -- both are
    # label 1 -- but if the score happened to separate them, a second threshold could
    # distinguish "treat now" from "old scarring". Measured among TB positives only.
    print("\ncan the TB score separate active from latent? (among TB positives only)")
    sep = {}
    for split in ("development", "calibration"):
        g = c01[c01.split == split]
        if g.empty:
            continue
        cat = g.category.to_numpy()
        m = np.isin(cat, ("active_tb", "latent_tb"))
        if m.sum() < 10 or len(set(cat[m])) < 2:
            continue
        p = calibrated(emb[[idx[i] for i in g.image_id]])[m]
        y_active = (cat[m] == "active_tb").astype(int)
        a = float(SEL.auc(y_active, p))
        sep[split] = {"auc_active_vs_latent": round(a, 4),
                      "n_active": int(y_active.sum()),
                      "n_latent": int((1 - y_active).sum())}
        print("  %-13s AUC %.4f (active %d vs latent %d)%s"
              % (split, a, y_active.sum(), (1 - y_active).sum(),
                 "  <- indistinguishable" if 0.4 < a < 0.6 else ""))
    n_lat = sum(r["n"] for r in rows if r["subtype"] in ("latent_tb", "active_latent_tb"))
    n_tb = sum(r["n"] for r in rows if r["subtype"] == "ALL_TB_blended")
    print("  %d of %d TB-positive cases (%.0f%%) are latent or mixed, and the system flags"
          % (n_lat, n_tb, 100 * n_lat / max(n_tb, 1)))
    print("  them as TB with the same confidence as active disease.")

    out = {"created": time.strftime("%Y-%m-%d"), "t_lower": t_lower, "temperature": T,
           "active_vs_latent_separability": sep,
           "latent_share_of_tb_positives": round(n_lat / max(n_tb, 1), 4),
           "dev_auc": round(float(mean_auc), 4), "seeds": list(SEEDS),
           "by_subtype": rows, "pooled_active_vs_latent": pooled,
           "note": ("binary_labels maps every TB subtype to positive, so the headline "
                    "sensitivity blends active and latent disease. Provisional split; "
                    "internal_test untouched. Calibration was used to fit the thresholds, "
                    "so its rows are in-sample for t_lower.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
