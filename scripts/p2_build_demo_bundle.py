# -*- coding: utf-8 -*-
"""Build a self-contained inference bundle so the system can actually run on a new image.

Produces everything `p2_cls.pipeline.run_case` needs, from the real embeddings:

    artifacts/p2/bundle/
        head.npz          trained classifier head (frozen-backbone linear head)
        bank.npz          OOD reference bank (training embeddings, subsampled)
        bundle.json       variant, thresholds, calibration, provenance

Thresholds and calibration:
  - temperature is fitted on **development** (never calibration or test), per the
    methodology's ordering, so the conformal score is fixed before calibration is used;
  - the conformal lower threshold and the precision-constrained upper threshold are fitted
    on **calibration**;
  - the OOD threshold is conformal on development.

Ownership note: temperature and the triage thresholds are Person 4's decisions (D22/D23) in
the real system. They are computed here so the demo runs end to end; when P4's fitted values
exist, load those instead — the bundle format is the same.

    py -3 scripts/p2_build_demo_bundle.py
"""
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import heads as H                                        # noqa: E402
from p2_cls import registry as R                                     # noqa: E402
from p2_cls import selection as S                                    # noqa: E402
from p2_gate import ood as O                                         # noqa: E402

VARIANT = "full_clahe0"          # measured: no crop variant beats it (see Section 14 doc)
EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT = ROOT / "artifacts/p2/bundle"
BANK_N = 4000                    # subsample the bank to keep the bundle small and fast
ALPHA_FNR = 0.10                 # conformal miss-rate target for the lower threshold
ALPHA_OOD = 0.05                 # conformal false-flag target for the OOD gate
TARGET_PRECISION = 0.80
# D13 amendment: the largest threshold still catching 100% of both alien pools, less a
# margin. Set to None to fall back to the raw conformal threshold.
OOD_THRESHOLD_OVERRIDE = 0.70


def fit_temperature(logits, y):
    """Ternary search on NLL. Fitted on development only."""
    lo, hi = 0.2, 5.0
    def nll(T):
        z = np.clip(logits / T, -30.0, 30.0)     # clip BEFORE exp; avoids log(0)
        p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    for _ in range(80):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(a) < nll(b): hi = b
        else: lo = a
    return (lo + hi) / 2


def conformal_lower(tb_scores, alpha):
    """Largest t with (1 + #{s_i < t})/(n+1) <= alpha. Abstains below n = 1/alpha - 1."""
    s = np.sort(np.asarray(tb_scores, float))
    n = len(s)
    k = int(np.floor((n + 1) * alpha))
    if k < 1:
        return None
    return float(s[k - 1])


def precision_upper(p, y, t_lower, target=TARGET_PRECISION):
    for t in np.linspace(t_lower, 1.0, 1000):
        m = p >= t
        if m.sum() >= 10:
            prec = float(y[m].sum() / m.sum())
            if prec >= target:
                return float(t), prec
    return 1.01, 0.0


def main():
    import pandas as pd
    if not EMB.exists() or not SPLITS.exists():
        raise SystemExit("need real embeddings and the split file")
    OUT.mkdir(parents=True, exist_ok=True)

    with np.load(EMB) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]

    def take(split):
        g = c01[c01.split == split]
        return (emb[[idx[i] for i in g.image_id]],
                H.binary_labels(g.category.tolist()))

    Xtr, ytr = take("train")
    Xdv, ydv = take("development")
    Xcal, ycal = take("calibration")
    print("train %d | development %d | calibration %d (internal_test untouched)"
          % (len(ytr), len(ydv), len(ycal)))

    # 1. train the head (5 seeds, averaged)
    res, mean_auc, sd = H.train_seeds(
        Xtr, ytr, Xdv, ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=(17, 29, 43, 61, 79))
    print("head: dev AUC %.4f (seed sd %.4f)" % (mean_auc, sd))
    head_path = OUT / "head.npz"
    sha = R.save_head(res[0], head_path)
    entry = R.RegistryEntry(model_id="radino_%s_bin" % VARIANT, head_type="binary",
                            variant=VARIANT, backbone="microsoft/rad-dino@110cbc18",
                            weight_path=str(head_path), weight_sha256=sha,
                            n_classes=2, is_primary=True)

    # 2. temperature on DEVELOPMENT
    p_dv = R.classify_embeddings(entry, Xdv)
    logit_dv = np.log(np.clip(p_dv, 1e-6, 1 - 1e-6) / (1 - np.clip(p_dv, 1e-6, 1 - 1e-6)))
    T = fit_temperature(logit_dv, ydv.astype(float))
    print("temperature (development): %.4f" % T)

    def calibrated(p):
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1 / (1 + np.exp(-lg / T))

    # 3. triage thresholds on CALIBRATION
    p_cal = calibrated(R.classify_embeddings(entry, Xcal))
    t_lower = conformal_lower(p_cal[ycal == 1], ALPHA_FNR)
    t_upper, prec = precision_upper(p_cal, ycal, t_lower or 0.0)
    print("t_lower (conformal, alpha=%.2f): %s | t_upper (precision>=%.2f): %.4f (prec %.3f)"
          % (ALPHA_FNR, "%.4f" % t_lower if t_lower else "ABSTAIN", TARGET_PRECISION, t_upper, prec))

    # 4. OOD bank + OOD threshold
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Xtr), min(BANK_N, len(Xtr)), replace=False)
    bank = O.l2_normalise(Xtr[sel])
    bank_obj = O.OodBank(bank)
    ood_fit = O.fit(bank_obj, Xdv, alpha=ALPHA_OOD)
    print("OOD threshold (conformal, alpha=%.2f): %s"
          % (ALPHA_OOD, "%.4f" % ood_fit.threshold if ood_fit.threshold else "ABSTAIN"))
    # The conformal threshold places itself at the 95th percentile of in-distribution
    # scores, so it *produces* a 5% false-flag rate whether or not any alien input sits
    # near that boundary - and it does not. Measured, it discards 9.67% of sick frontals
    # and 3.39% of TB cases while alien detection is already saturated at 100% far above
    # it. See docs/p2_ood_threshold_is_miscalibrated.md.
    ood_threshold = ood_fit.threshold
    ood_rule = "conformal_alpha_%.2f" % ALPHA_OOD
    if ood_threshold is not None and OOD_THRESHOLD_OVERRIDE is not None:
        print("  overriding with the alien-margin threshold %.4f (D13 amendment)"
              % OOD_THRESHOLD_OVERRIDE)
        ood_threshold = OOD_THRESHOLD_OVERRIDE
        ood_rule = "alien_margin_100pct_both_pools"
    with open(OUT / "bank.npz", "wb") as fh:
        np.savez_compressed(fh, bank=bank.astype(np.float16))

    # sanity: what does this bundle do on development?
    p_dv_cal = calibrated(p_dv)
    cat = np.where(p_dv_cal >= t_upper, "TB",
                   np.where(p_dv_cal < (t_lower or 0), "NOT_TB", "REFER"))
    screen_pos = (cat != "NOT_TB")
    sens = float(screen_pos[ydv == 1].mean())
    spec = float((~screen_pos)[ydv == 0].mean())
    print("development behaviour: sensitivity %.3f | specificity %.3f | TB %d REFER %d NOT_TB %d"
          % (sens, spec, (cat == "TB").sum(), (cat == "REFER").sum(), (cat == "NOT_TB").sum()))

    bundle = {
        "created": str(date.today()), "variant": VARIANT,
        "backbone": "microsoft/rad-dino@110cbc18d5133582e320b43d53bf5c44e410c936",
        "model": entry.to_json(),
        "temperature": T, "temperature_fitted_on": "development",
        "t_lower": t_lower, "t_upper": t_upper, "alpha_fnr": ALPHA_FNR,
        "target_precision": TARGET_PRECISION, "achieved_precision": prec,
        "thresholds_fitted_on": "calibration",
        "ood_threshold": ood_threshold, "ood_alpha": ALPHA_OOD,
        "ood_threshold_rule": ood_rule,
        "ood_threshold_conformal": ood_fit.threshold,
        "ood_k": bank_obj.k, "bank_n": int(len(bank)),
        "dev_auc": round(mean_auc, 4), "dev_sensitivity": round(sens, 4),
        "dev_specificity": round(spec, 4),
        "provisional": True,
        "note": ("Provisional: split is not Person 1's C01, and temperature/triage "
                 "thresholds are Person 4's decisions (D22/D23) computed here so the demo "
                 "runs end to end. internal_test was never read."),
    }
    json.dump(bundle, open(OUT / "bundle.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % (OUT / "bundle.json"))


if __name__ == "__main__":
    sys.exit(main())
