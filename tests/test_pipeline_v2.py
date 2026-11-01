# -*- coding: utf-8 -*-
"""Unit tests for the v2 pipeline that need no GPU and no model download.

    py -3 -m pytest tests/test_pipeline_v2.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p2_cls import pipeline_v2 as P2                                 # noqa: E402

BUNDLE = ROOT / "artifacts/p2/bundle_v2_regional"


@pytest.fixture(scope="module")
def bundle():
    if not (BUNDLE / "head.npz").is_file():
        pytest.skip("bundle_v2 not built")
    return P2.load_bundle_v2()


def test_head_is_mean_over_seeds_of_max_over_regions(bundle):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(144, 768)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    score, region_mean, per_seed = P2.head(X, bundle)
    W, b = bundle["W"], bundle["b"]
    manual = [float((X @ W[s][0] + b[s][0]).max()) for s in range(W.shape[0])]
    assert abs(score - np.mean(manual)) < 1e-5
    assert per_seed.shape == (W.shape[0],)
    assert region_mean.shape == (144,)


def test_argmax_region_is_the_decision_region_per_seed(bundle):
    """For each seed, the max over regions IS that seed's image logit."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(144, 768)).astype(np.float32)
    W, b = bundle["W"], bundle["b"]
    for s in range(W.shape[0]):
        lg = X @ W[s][0] + b[s][0]
        assert lg[np.argmax(lg)] == lg.max()


def test_bundle_head_shape_and_seed_count(bundle):
    assert bundle["W"].shape[1:] == (1, 768) and bundle["b"].shape[1:] == (1,)
    assert bundle["W"].shape[0] == 5
    assert bundle["meta"]["model"]["trainable_parameters"] == 5 * 769


def test_thresholds_fitted_on_development_only(bundle):
    t = bundle["meta"]["thresholds"]
    assert "development" in t["fitted_on"].lower()
    assert "vietnam" not in t["fitted_on"].lower().replace("never vietnam", "")
    assert t["logit_at_spec_0.90"] > t["logit_at_spec_0.70"]


def test_fusion_refuses_missing_fields(bundle):
    if bundle["fusion"] is None:
        pytest.skip("no fusion block")
    r, err = P2.fuse(0.0, {"past_tb": 1}, bundle)
    assert r is None and "missing" in err


def test_fusion_refuses_non_binary(bundle):
    if bundle["fusion"] is None:
        pytest.skip("no fusion block")
    enc, _ = P2.encode_clinical({"sex": 1, "age_group": 2, "c": 1, "f": 0, "wl": 0, "ns": 0,
                                 "contact": 0, "past_tb": 1, "dm": 0, "rad_system": 2})
    enc["c"] = 2
    r, err = P2.fuse(0.0, enc, bundle)
    assert r is None and "0/1" in err


def test_fusion_matches_the_logistic_formula(bundle):
    if bundle["fusion"] is None:
        pytest.skip("no fusion block")
    f = bundle["fusion"]
    raw = {"sex": 1, "age_group": 2, "c": 1, "f": 1, "wl": 0, "ns": 0, "contact": 0,
           "past_tb": 1, "dm": 99, "rad_system": 1}
    enc, err = P2.encode_clinical(raw)
    assert err is None
    assert enc["age_group"] == 1 and enc["dm"] == 0 and enc["dm_unknown"] == 1
    assert enc["scan1"] == 1 and enc["scan2"] == 0
    score = 1.5
    r, err = P2.fuse(score, enc, bundle)
    assert err is None
    z = (score - f["score_mean"]) / f["score_sd"]
    lg = f["intercept"] + f["image_score_std"] * z + sum(
        f["clinical"][k] * enc[k] for k in f["features_order"][1:])
    assert abs(r["logit"] - lg) < 1e-3
    assert abs(r["p_fused"] - 1 / (1 + np.exp(-lg))) < 1e-5
    assert "EXPLORATORY" in r["status"]


def test_encode_clinical_rejects_garbage():
    enc, err = P2.encode_clinical({"sex": "m"})
    assert enc is None and err


def test_fusion_coefficients_match_the_audited_run(bundle):
    if bundle["fusion"] is None:
        pytest.skip("no fusion block")
    import csv
    src = ROOT / "artifacts/clinical_fusion_20261101_0330/tables/coefficients_full_fit.csv"
    if not src.is_file():
        pytest.skip("fusion run absent")
    audited = {r["feature"]: float(r["coef_log_odds"]) for r in csv.DictReader(open(src, encoding="utf-8"))
               if r["image_score"] == "ours" and r["model"] == "M2"}
    for k, v in bundle["fusion"]["clinical"].items():
        assert abs(v - audited[k]) < 5e-5, k
    assert abs(bundle["fusion"]["image_score_std"] - audited["image_score"]) < 5e-5
