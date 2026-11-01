# -*- coding: utf-8 -*-
"""Freeze research candidate v3: RAD-DINO-MAIRA-2 encoder, active-TB target, regional max
head, optional 12-field clinical fusion. NOT promoted; v2 stays the default.

Head weights are the five seeds fitted and gated in R22
(artifacts/fusion_on_stacked_head_20261101_0800/tables/X_noNLM_head.npz) - copied, not
retrained, so the bundle is the audited model. Thresholds come from TBX11K development only.
Fusion coefficients are a full-data refit on Vietnam TUNE (exposed) of the R22 model.

    py -3 scripts/p2_39_build_bundle_v3.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_cls import selection as SEL                                 # noqa: E402

R20 = ROOT / "artifacts/multiencoder_ensemble_20261101_0600"
R22 = ROOT / "artifacts/fusion_on_stacked_head_20261101_0800"
R23 = ROOT / "artifacts/clinical_field_ablation_20261101_0900"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
VSPLIT = ROOT / "artifacts/p2/external/vietnam_tune_report_split.json"
VCSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
OUT = ROOT / "artifacts/p2/bundle_v3_candidate"
BACKBONE = "microsoft/rad-dino-maira-2@a6132efb9268859545e9f68be18a76a6473f174a"
SEEDS = (17, 29, 43, 61, 79)
TB = ("active_tb", "latent_tb", "active_latent_tb")
ACTIVE = ("active_tb", "active_latent_tb")
CLIN = ["sex", "age_group", "c", "f", "wl", "ns", "contact", "past_tb", "dm", "dm_unknown", "scan1", "scan2"]
GATE = {"tune_raw": 0.7585, "cal_vs_active": 0.9466, "fused_cv": 0.7941}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_cache(path, ids):
    with np.load(path) as z:
        zi, X = z["image_id"].tolist(), z["reg"]
    idx = {i: k for k, i in enumerate(zi)}
    return X[[idx[i] for i in ids]]


def score(X, W, b):
    return np.mean([(X.astype(np.float32) @ W[s].T + b[s])[:, :, 0].max(1) for s in range(W.shape[0])], 0)


def main():
    import pandas as pd
    t0 = time.perf_counter()
    with np.load(R22 / "tables/X_noNLM_head.npz") as z:
        W, b = z["W"].astype(np.float32), z["b"].astype(np.float32)
    assert W.shape == (5, 1, 768) and b.shape == (5, 1)

    c01 = pd.read_csv(C01)
    dv = c01[c01.split == "development"]
    cal = c01[c01.split == "calibration"]
    Xdv = load_cache(R20 / "cache/regional_D2_tbx_dev_lungs_g12.npz", dv.image_id)
    Xcal = load_cache(R20 / "cache/regional_D2_tbx_cal_lungs_g12.npz", cal.image_id)
    y_dv_act = np.array([1 if c in ACTIVE else 0 for c in dv.category])
    y_dv_tb = np.array([1 if c in TB else 0 for c in dv.category])
    y_cal = np.array([1 if c in ACTIVE else 0 for c in cal.category])
    p_dev, p_cal = score(Xdv, W, b), score(Xcal, W, b)
    dev_auc_act, dev_auc_tb = float(SEL.auc(y_dv_act, p_dev)), float(SEL.auc(y_dv_tb, p_dev))
    cal_auc = float(SEL.auc(y_cal, p_cal))
    thr70 = float(np.quantile(p_dev[y_dv_act == 0], 0.70))
    thr90 = float(np.quantile(p_dev[y_dv_act == 0], 0.90))
    print("dev AUC vs active %.4f (vs tb_label %.4f) | cal vs active %.4f | thr 0.70 %.4f, 0.90 %.4f"
          % (dev_auc_act, dev_auc_tb, cal_auc, thr70, thr90))
    assert abs(cal_auc - GATE["cal_vs_active"]) < 5e-4, "calibration gate"

    # fusion: full-data refit of the R22 M2 model on the TUNE X scores
    sp = json.loads(VSPLIT.read_text(encoding="utf-8"))
    tune = {str(i) for i in sp["tune"]}
    m = pd.read_csv(VCSV)
    m["pid"] = m.pid.astype(str)
    m = m[m.pid.isin(tune)].sort_values("pid").reset_index(drop=True)
    assert len(m) == 1352
    s = score(load_cache(R20 / "cache/regional_D2_vie_tune_lungs_g12.npz", m.pid.tolist()), W, b)
    y = m.xpert_result.to_numpy(int)
    tune_auc = float(SEL.auc(y, s))
    print("TUNE raw AUROC %.4f" % tune_auc)
    assert abs(tune_auc - GATE["tune_raw"]) < 5e-4, "TUNE gate"
    mu, sd = float(s.mean()), float(s.std())
    X = np.column_stack([m.sex, (m.age_group == 2).astype(float), m.c, m.f, m.wl, m.ns, m.contact,
                         m.past_tb, (m.dm == 1).astype(float), (m.dm == 99).astype(float),
                         (m.rad_system == 1).astype(float), (m.rad_system == 2).astype(float)]).astype(float)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([(s - mu) / sd, X]), y)
    clinical = {k: float(v) for k, v in zip(CLIN, lr.coef_[0][1:])}

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "head.npz", W=W, b=b, seeds=np.array(SEEDS))
    (OUT / "fusion.json").write_text(json.dumps({
        "status": "EXPLORATORY - fitted on Vietnam TUNE (exposed); optional at inference; "
                  "scanner columns are site calibration, not clinical context (R23)",
        "source_run": str(R22.relative_to(ROOT)).replace("\\", "/"),
        "model": "logistic regression, L2, C=1.0, on standardised image score + 12 fields",
        "cv_result": {"auroc_image_only": 0.7585, "auroc_fused": 0.7941, "delta": 0.0365, "ci95": [0.0154, 0.0576],
                      "prev_tb_auroc_image_only": 0.6501, "prev_tb_auroc_fused": 0.7349,
                      "vs_comparator_raw_0.7780": {"delta": 0.0170, "ci95": [-0.0193, 0.0527]}},
        "field_decomposition_R23": {
            "questionnaire_only_10_fields": 0.7803, "image_plus_scanner_only": 0.7788,
            "cost_of_dropping_scanner": {"delta": -0.0138, "ci95": [-0.0250, -0.0029], "prev_tb_delta": -0.0429},
            "note": "the scanner term is fitted to the two Dryad scanners and has no meaning at another site; "
                    "a new site needs its own term fitted on local labelled data"},
        "features_order": ["image_score_std"] + CLIN,
        "feature_definitions": {
            "sex": "0/1 as in the Dryad CSV", "age_group": "1 if Dryad age_group == 2 else 0",
            "c": "cough 0/1", "f": "fever 0/1", "wl": "weight loss 0/1", "ns": "night sweats 0/1",
            "contact": "TB contact 0/1", "past_tb": "self-reported previous TB 0/1",
            "dm": "1 if diabetes == 1 else 0", "dm_unknown": "1 if diabetes field == 99 else 0",
            "scan1": "1 if scanner == 1", "scan2": "1 if scanner == 2 (scanner 3 is reference)"},
        "intercept": float(lr.intercept_[0]), "image_score_std": float(lr.coef_[0][0]),
        "score_mean": mu, "score_sd": sd, "clinical": clinical}, indent=1), encoding="utf-8")
    meta = {
        "created": time.strftime("%Y-%m-%d"), "bundle_version": 3,
        "status": "RESEARCH CANDIDATE v3 - NOT PROMOTED. v2 (bundle_v2_regional) remains the default.",
        "model": {"model_id": "maira2_regional_g12_max_active", "head_type": "regional_max",
                  "grid": 12, "aggregator": "max", "seeds": list(SEEDS),
                  "target": "active_tb_label (latent / obsolete TB labelled negative)",
                  "trainable_parameters": int(W.size + b.size),
                  "weight_path": "head.npz", "weight_sha256": sha(OUT / "head.npz"),
                  "weights_copied_from": "artifacts/fusion_on_stacked_head_20261101_0800/tables/X_noNLM_head.npz"},
        "backbone": BACKBONE,
        "features": "512px lungs-masked -> 518 -> 37x37 patches -> 12x12 adaptive-avg regions, L2 per region",
        "training_rows": {"split": "train", "n": 3241, "positives": 447,
                          "categories": list(TB) + ["healthy"], "sick_non_tb_excluded": True,
                          "nlm_included": False},
        "dev_auc": round(dev_auc_act, 4), "dev_auc_vs_tb_label": round(dev_auc_tb, 4),
        "thresholds": {"logit_at_spec_0.70": round(thr70, 4), "logit_at_spec_0.90": round(thr90, 4),
                       "fitted_on": "TBX11K development (659), negatives = not active TB - never Vietnam, never calibration"},
        "known_external": {"tbx11k_calibration_auroc_vs_active": round(cal_auc, 4),
                           "vietnam_tune_auroc_raw": round(tune_auc, 4),
                           "vietnam_tune_auroc_fused_cv": 0.7941,
                           "vietnam_tune_auroc_fused_no_scanner_cv": 0.7803,
                           "comparator_vietnam_tune_raw": 0.7780,
                           "note": "TUNE is exposed: encoder, target and fusion were all chosen on it (R20-R23). "
                                   "Not validation. Vietnam REPORT is also exposed (six scoring events, R7) "
                                   "and is not read here."},
        "provenance_runs": ["multiencoder_ensemble_20261101_0600", "stacked_levers_20261101_0700",
                            "fusion_on_stacked_head_20261101_0800", "clinical_field_ablation_20261101_0900"],
        "fusion": "fusion.json (optional; EXPLORATORY)",
        "attribution": "image logit = max over 144 regions of the region logit; the arg-max region is exactly the decision region",
        "provisional": True,
        "note": "Candidate for evaluation on an untouched cohort against the comparator, previous-TB as a "
                "pre-specified subgroup. Split is the provisional C01; internal_test never read.",
    }
    (OUT / "bundle.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote %s (head %s) in %.1f s" % (OUT, meta["model"]["weight_sha256"][:16], time.perf_counter() - t0))


if __name__ == "__main__":
    main()
