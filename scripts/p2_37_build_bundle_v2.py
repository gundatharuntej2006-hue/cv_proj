# -*- coding: utf-8 -*-
"""Build the v2 inference bundle: the regional 12x12 max head, plus optional clinical fusion.

The served bundle in artifacts/p2/bundle/ is the 2026-10-14 global head
(radino_full_clahe0_bin, dev AUROC 0.9949, Vietnam full-cohort 0.6744). Every result since
2026-10-20 is about the regional head (TBX11K dev 0.9368, Vietnam REPORT 0.7242). This
writes that head as a separate bundle so the serving layer can run the model the research
is actually about. The v1 bundle is left untouched.

    py -3 scripts/p2_37_build_bundle_v2.py          # -> artifacts/p2/bundle_v2_regional/

Contents:
  head.npz        W [5,1,768], b [5,1]  - the five production seeds; score = mean over seeds
                  of max over regions of (x . W_s + b_s)
  bundle.json     provenance, thresholds (TBX11K development only), fusion block
  fusion.json     logistic-regression coefficients from clinical_fusion_20261101_0330,
                  full-data fit on Vietnam TUNE - EXPLORATORY, optional at inference

Nothing here is read from Vietnam REPORT or TBX11K internal_test.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import p2_32_mil_sweep as S                                         # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

OUT = ROOT / "artifacts/p2/bundle_v2_regional"
REG = ROOT / "artifacts/p2/regional"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
FUSION = ROOT / "artifacts/clinical_fusion_20261101_0330"
PRED = ROOT / "artifacts/active_tb_label_raydino_20261024_1254/outputs/predictions/vietnam_tune_armA.csv"
SEEDS = (17, 29, 43, 61, 79)
TB = ("active_tb", "latent_tb", "active_latent_tb")
CLIN = ["sex", "age_group", "c", "f", "wl", "ns", "contact", "past_tb", "dm", "dm_unknown",
        "scan1", "scan2"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import pandas as pd
    t0 = time.perf_counter()
    with np.load(REG / "regional_tbx11k_lungs_g12_v1.npz") as z:
        tid, TX = z["image_id"].tolist(), z["reg"]
    ti = {i: k for k, i in enumerate(tid)}
    c01 = pd.read_csv(C01)
    assert not (set(c01[c01.split == "internal_test"].image_id) & set(tid))
    c01 = c01[c01.image_id.isin(ti)]
    tr = c01[(c01.split == "train") & c01.category.isin(TB + ("healthy",))]
    dv = c01[c01.split == "development"]
    Xtr = S.repool(TX[[ti[i] for i in tr.image_id]], 12, 12)
    Xdv = S.repool(TX[[ti[i] for i in dv.image_id]], 12, 12)
    ytr = np.array([1 if c in TB else 0 for c in tr.category])
    ydv = np.array([1 if c in TB else 0 for c in dv.category])

    Ws, bs, devs = [], [], []
    for sd in SEEDS:
        W, b, best = S.train(Xtr, ytr, Xdv, ydv, "max", sd)
        Ws.append(W); bs.append(b); devs.append(best)
        print("  seed %2d dev AUC %.4f" % (sd, best))
    W = np.stack(Ws).astype(np.float32)          # [5,1,768]
    b = np.stack(bs).astype(np.float32)          # [5,1]
    p_dev = np.mean([S.score(Xdv, W[i], b[i], "max") for i in range(len(SEEDS))], 0)
    dev_auc = float(SEL.auc(ydv, p_dev))
    thr70 = float(np.quantile(p_dev[ydv == 0], 0.70))
    thr90 = float(np.quantile(p_dev[ydv == 0], 0.90))
    print("dev AUC (seed mean) %.4f | thresholds: 0.70-spec %.4f, 0.90-spec %.4f" % (dev_auc, thr70, thr90))

    # fusion: full-data coefficients from the CV run, plus the score standardisation
    coefs = {}
    for r in csv.DictReader(open(FUSION / "tables/coefficients_full_fit.csv", encoding="utf-8")):
        if r["image_score"] == "ours" and r["model"] == "M2":
            coefs[r["feature"]] = float(r["coef_log_odds"])
    with open(PRED, encoding="utf-8") as fh:
        s_tune = np.array([float(r["score_seed_mean"]) for r in csv.DictReader(fh)])
    # the LR was fitted without an exported intercept; refit it exactly to recover it
    from sklearn.linear_model import LogisticRegression
    m = pd.read_csv(ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv")
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    m["pid"] = m.pid.astype(str)
    m = m[m.pid.isin({str(i) for i in sp["tune"]})].sort_values("pid").reset_index(drop=True)
    assert len(m) == 1352
    with open(PRED, encoding="utf-8") as fh:
        d = {r["pid"]: float(r["score_seed_mean"]) for r in csv.DictReader(fh)}
    s = np.array([d[p] for p in m.pid])
    mu, sd = float(s.mean()), float(s.std())
    X = np.column_stack([m.sex, (m.age_group == 2).astype(float), m.c, m.f, m.wl, m.ns, m.contact,
                         m.past_tb, (m.dm == 1).astype(float), (m.dm == 99).astype(float),
                         (m.rad_system == 1).astype(float), (m.rad_system == 2).astype(float)]).astype(float)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([(s - mu) / sd, X]), m.xpert_result)
    fused = {"intercept": float(lr.intercept_[0]), "image_score_std": float(lr.coef_[0][0]),
             "score_mean": mu, "score_sd": sd,
             "clinical": {k: float(v) for k, v in zip(CLIN, lr.coef_[0][1:])}}
    for k, v in fused["clinical"].items():
        assert abs(v - coefs[k]) < 5e-5, k          # audited CSV is rounded to 4 dp

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "head.npz", W=W, b=b, seeds=np.array(SEEDS))
    (OUT / "fusion.json").write_text(json.dumps({
        "status": "EXPLORATORY - fitted on Vietnam TUNE (exposed); optional at inference",
        "source_run": "artifacts/clinical_fusion_20261101_0330",
        "model": "logistic regression, L2, C=1.0, on standardised image score + 12 clinical fields",
        "cv_result": {"auroc_image_only": 0.7237, "auroc_fused": 0.7809,
                      "delta": 0.058, "ci95": [0.030, 0.085],
                      "prev_tb_auroc_image_only": 0.6144, "prev_tb_auroc_fused": 0.7122},
        "features_order": ["image_score_std"] + CLIN,
        "feature_definitions": {
            "sex": "0/1 as in the Dryad CSV", "age_group": "1 if Dryad age_group == 2 else 0",
            "c": "cough 0/1", "f": "fever 0/1", "wl": "weight loss 0/1", "ns": "night sweats 0/1",
            "contact": "TB contact 0/1", "past_tb": "self-reported previous TB 0/1",
            "dm": "1 if diabetes == 1 else 0", "dm_unknown": "1 if diabetes field == 99 else 0",
            "scan1": "1 if scanner == 1", "scan2": "1 if scanner == 2 (scanner 3 is reference)"},
        **fused}, indent=1), encoding="utf-8")
    meta = {
        "created": time.strftime("%Y-%m-%d"), "bundle_version": 2,
        "model": {"model_id": "radino_regional_g12_max_bin", "head_type": "regional_max",
                  "grid": 12, "aggregator": "max", "seeds": list(SEEDS),
                  "trainable_parameters": int(W.size + b.size),
                  "weight_path": "head.npz", "weight_sha256": sha(OUT / "head.npz")},
        "backbone": "microsoft/rad-dino@110cbc18d5133582e320b43d53bf5c44e410c936",
        "features": "512px lungs-masked -> 518 -> 37x37 patches -> 12x12 adaptive-avg regions, L2 per region",
        "training_rows": {"split": "train", "n": int(len(ytr)), "positives": int(ytr.sum()),
                          "categories": list(TB) + ["healthy"], "sick_non_tb_excluded": True},
        "dev_auc": round(dev_auc, 4), "dev_auc_per_seed": [round(x, 4) for x in devs],
        "thresholds": {"logit_at_spec_0.70": round(thr70, 4), "logit_at_spec_0.90": round(thr90, 4),
                       "fitted_on": "TBX11K development (659) - never Vietnam, never calibration"},
        "known_external": {"vietnam_report_auroc": 0.7242, "vietnam_tune_auroc": 0.7237,
                           "shenzhen_auroc": 0.969, "montgomery_auroc": 0.969,
                           "note": "from mil_final.json and p2/audit/cross_dataset_metrics.csv"},
        "fusion": "fusion.json (optional; EXPLORATORY)",
        "attribution": "image logit = max over 144 regions of the region logit; the arg-max region is exactly the decision region",
        "provisional": True,
        "note": ("v2 bundle. Split is the provisional C01; internal_test never read; Vietnam "
                 "REPORT never read for any threshold or coefficient. Not clinically validated."),
    }
    (OUT / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote %s (head %s) in %.1f s" % (OUT, meta["model"]["weight_sha256"][:16], time.perf_counter() - t0))


if __name__ == "__main__":
    main()
