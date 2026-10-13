# -*- coding: utf-8 -*-
"""P2-08 / P2-09: preprocessing and head selection. Tested on the mock C08, which carries
five models spanning CLAHE on/off, crop/full and binary/three-class — exactly the choices
these tasks decide."""
import math
from pathlib import Path

import numpy as np
import pytest

from p2_cls import selection as S


# ---------------------------------------------------------------- the metric
def test_sensitivity_at_specificity_is_exact_on_a_clean_split():
    # negatives 0..0.69, positives 0.70..1.0 -> perfectly separable
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    p = np.r_[np.linspace(0, 0.69, 100), np.linspace(0.70, 1.0, 100)]
    sens, thr, n_pos, n_tp = S.sensitivity_at_specificity(y, p, target_spec=0.70)
    assert sens == 1.0 and n_pos == 100 and n_tp == 100
    assert 0.45 < thr < 0.72


def test_sensitivity_drops_as_required_specificity_rises():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    p = np.r_[rng.normal(0.4, 0.15, 500), rng.normal(0.6, 0.15, 500)].clip(0, 1)
    s70, _, _, _ = S.sensitivity_at_specificity(y, p, 0.70)
    s95, _, _, _ = S.sensitivity_at_specificity(y, p, 0.95)
    assert s70 >= s95                       # demanding more specificity costs sensitivity


def test_wilson_ci_matches_the_known_worked_example():
    lo, hi = S.wilson_ci(93, 101)
    assert abs(lo - 0.8514) < 1e-3 and abs(hi - 0.9593) < 1e-3


def test_wilson_agrees_with_p4_stats_if_available():
    try:
        from p4_stats.ci import wilson_score_interval
    except Exception:
        pytest.skip("p4_stats not in this repo")
    for k, n in [(50, 100), (9, 10), (1, 200)]:
        a = S.wilson_ci(k, n)
        b = wilson_score_interval(k, n)
        assert np.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------- ranking rule
def _cand(mid, variant, head, clahe, margin, sens, auc):
    return S.Candidate(mid, variant, head, clahe, margin, sens, sens - 0.03, sens + 0.03,
                       auc, 100, 1000)


def test_higher_sensitivity_wins():
    a = _cand("a", "v", "binary", False, 0.10, 0.80, 0.90)
    b = _cand("b", "v", "binary", False, 0.10, 0.70, 0.99)
    assert a.better_than(b) and not b.better_than(a)


def test_auc_breaks_a_sensitivity_tie():
    a = _cand("a", "v", "binary", False, 0.10, 0.802, 0.95)
    b = _cand("b", "v", "binary", False, 0.10, 0.800, 0.91)
    assert a.better_than(b)          # within 0.5pt sensitivity -> AUC decides


# ---------------------------------------------------------------- on the mock C08
@pytest.fixture
def mock_c08():
    import pyarrow.parquet as pq
    p = Path(__file__).resolve().parents[1] / "mock/artifacts/p2/scores/scores_tbx11k_development_v1.parquet"
    if not p.exists():
        pytest.skip("run scripts/p2_02_make_mocks.py first")
    return pq.read_table(p).to_pandas()


def label_fn(image_id):
    return 1 if str(image_id).startswith("tb") else 0


def test_candidate_table_covers_every_model(mock_c08):
    cands = S.candidate_table(mock_c08, label_fn)
    assert len(cands) == mock_c08.model_id.nunique()
    for c in cands:
        assert 0.0 <= c.sensitivity <= 1.0
        assert c.sens_ci_low <= c.sensitivity <= c.sens_ci_high
        assert c.n_pos > 0


def test_table_is_sorted_best_first(mock_c08):
    cands = S.candidate_table(mock_c08, label_fn)
    sens = [c.sensitivity for c in cands]
    assert sens == sorted(sens, reverse=True)


def test_preprocessing_selection_returns_a_valid_choice(mock_c08):
    cands = S.candidate_table(mock_c08, label_fn)
    d08_09 = S.select_preprocessing(cands)
    assert isinstance(d08_09["clahe"], bool)
    assert d08_09["margin"] in (None, 0.05, 0.10, 0.15)
    assert d08_09["model_id"].startswith("radino")
    # the chosen RAD-DINO binary model must be the best among RAD-DINO binaries
    radino_bin = [c for c in cands if c.head == "binary" and c.model_id.startswith("radino")]
    assert d08_09["sensitivity"] == max(c.sensitivity for c in radino_bin)


def test_head_selection_names_primary_and_auxiliary(mock_c08):
    cands = S.candidate_table(mock_c08, label_fn)
    variant = S.select_preprocessing(cands)["variant"]
    d11 = S.select_head(cands, "crop_m0.10_clahe0")   # the variant with both heads in the mock
    assert d11["primary_head"] in ("binary", "three_class")
    if d11.get("auxiliary_head"):
        assert d11["auxiliary_head"] != d11["primary_head"]
        assert "binary" in d11 and "three_class" in d11


def test_selection_is_deterministic(mock_c08):
    a = S.select_preprocessing(S.candidate_table(mock_c08, label_fn))
    b = S.select_preprocessing(S.candidate_table(mock_c08, label_fn))
    assert a == b
