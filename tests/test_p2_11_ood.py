# -*- coding: utf-8 -*-
"""P2-11: the kNN OOD gate. The tests that matter are the conformal guarantee and the
abstention below the minimum sample size — the property the throwaway prototype lacked and
the one that was wrong in Person 4's conformal code."""
import numpy as np
import pytest

from p2_gate import ood as O


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def gaussian_bank(rng, n=800, dim=16, direction=None, spread=0.10):
    """A cloud clustered around one DIRECTION (not magnitude): cosine distance only sees
    direction, so in-distribution means 'points the same way', which is how the real
    embeddings behave — a shared 'chest X-ray' direction with per-image scatter. The
    direction must dominate the scatter (small spread, modest dim) or random high-dim noise
    makes every pair near-orthogonal and the geometry stops resembling embeddings."""
    if direction is None:
        direction = np.zeros(dim)
        direction[0] = 1.0
    direction = direction / np.linalg.norm(direction)
    x = direction[None, :] + spread * rng.normal(size=(n, dim))
    return x


# --------------------------------------------------------------- scoring
def test_score_is_near_zero_for_bank_members(rng):
    emb = gaussian_bank(rng)
    bank = O.OodBank(emb)
    s = bank.score(emb, exclude_self=True)
    # an image's own near-neighbours in the same tight cluster are close
    assert np.median(s) < 0.2


def test_ood_cloud_scores_higher_than_in_distribution(rng):
    d_in = np.zeros(16); d_in[0] = 1.0
    d_ood = np.zeros(16); d_ood[1] = 1.0                    # an orthogonal direction
    bank = O.OodBank(gaussian_bank(rng, direction=d_in))
    ind = gaussian_bank(rng, n=200, direction=d_in)
    ood = gaussian_bank(rng, n=200, direction=d_ood)
    assert bank.score(ood).mean() > bank.score(ind).mean()
    assert bank.score(ood).mean() > 2 * bank.score(ind).mean()


def test_scores_are_l2_normalised_regardless_of_input_scale(rng):
    emb = gaussian_bank(rng)
    bank = O.OodBank(emb)
    q = gaussian_bank(rng, n=50)
    assert np.allclose(bank.score(q), bank.score(q * 7.3), atol=1e-5)


def test_bank_smaller_than_k_is_refused(rng):
    with pytest.raises(ValueError, match="k="):
        O.OodBank(gaussian_bank(rng, n=3), k=5)


# --------------------------------------------------------------- the conformal threshold
def test_minimum_n_matches_the_formula():
    assert O.min_calibration_n(0.10) == 9
    assert O.min_calibration_n(0.05) == 19
    assert O.min_calibration_n(0.01) == 99


def test_threshold_abstains_below_the_minimum(rng):
    scores = rng.random(8)
    assert O.fit_threshold(scores, alpha=0.10) is None      # need 9
    assert O.fit_threshold(rng.random(9), alpha=0.10) is not None


def test_threshold_abstains_on_empty_calibration():
    assert O.fit_threshold(np.array([]), alpha=0.10) is None


def test_conformal_guarantee_holds_empirically(rng):
    """The promise: P(flag a genuine in-distribution image) <= alpha. Check it over many
    resamples, which is the whole reason to prefer this over a plain percentile."""
    bank = O.OodBank(gaussian_bank(rng, n=1000))
    for alpha in (0.10, 0.05):
        flagged = []
        for _ in range(300):
            calib = gaussian_bank(rng, n=100)
            test = gaussian_bank(rng, n=200)
            t = O.fit_threshold(bank.score(calib), alpha=alpha)
            flagged.append((bank.score(test) > t).mean())
        assert np.mean(flagged) <= alpha + 0.01, (alpha, np.mean(flagged))


def test_fit_reports_abstention_with_a_reason(rng):
    bank = O.OodBank(gaussian_bank(rng))
    fit = O.fit(bank, gaussian_bank(rng, n=5), alpha=0.10)
    assert fit.abstained and fit.threshold is None
    assert "minimum" in fit.reason and "abstains" in fit.reason


def test_fit_succeeds_above_the_minimum(rng):
    bank = O.OodBank(gaussian_bank(rng))
    fit = O.fit(bank, gaussian_bank(rng, n=50), alpha=0.10)
    assert not fit.abstained and fit.threshold is not None
    assert fit.n_calibration == 50 and fit.k == 5


# --------------------------------------------------------------- is_ood / abstain behaviour
def test_is_ood_flags_above_threshold(rng):
    s = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    assert list(O.is_ood(s, 0.6)) == [False, False, True]


def test_an_abstained_gate_flags_nothing(rng):
    """A None threshold must never be read as 'reject everything' — that was the deployment
    failure the finding is about."""
    s = np.array([0.1, 0.5, 0.9, 2.0], dtype=np.float32)
    assert not O.is_ood(s, None).any()


def test_false_flag_rate_respects_the_guarantee(rng):
    bank = O.OodBank(gaussian_bank(rng, n=1000))
    t = O.fit_threshold(bank.score(gaussian_bank(rng, n=200)), alpha=0.10)
    ffr = O.false_flag_rate(bank, gaussian_bank(rng, n=500), t)
    assert ffr <= 0.15                       # alpha plus finite-sample slack


def test_false_flag_rate_is_zero_when_abstained(rng):
    bank = O.OodBank(gaussian_bank(rng))
    assert O.false_flag_rate(bank, gaussian_bank(rng, n=100), None) == 0.0


# --------------------------------------------------------------- on real embeddings
def test_on_real_embeddings_reproduces_the_finding():
    """The headline result, as a regression test: on the real embeddings a TBX11K-fitted
    threshold catches true OOD but the score still ranks a new site correctly."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    emb_dir = root / "artifacts/p2/emb"
    need = ["radino_tbx11k_full_clahe0_v1.npz", "radino_ood_lateral_v1.npz",
            "radino_ood_nonmedical_v1.npz"]
    if not all((emb_dir / n).exists() for n in need):
        pytest.skip("real embeddings not present")
    import pandas as pd

    def load(p):
        with np.load(emb_dir / p) as z:
            return z["image_id"].tolist(), z["emb"].astype(np.float32)
    ids, tb = load(need[0])
    _, lat = load(need[1])
    _, nm = load(need[2])
    sp = pd.read_csv(root / "artifacts/p2/dedup/provisional_official_splits.csv")
    split = dict(zip(sp.image_id, sp.split))
    bank_emb = tb[np.array([split.get(i) == "train" for i in ids])]
    held = tb[np.array([split.get(i) == "val" for i in ids])]

    bank = O.OodBank(bank_emb)
    t = O.fit_threshold(bank.score(held[:500]), alpha=0.05)
    assert t is not None
    # true OOD is caught
    assert (bank.score(lat) > t).mean() > 0.95
    assert (bank.score(nm) > t).mean() > 0.95
    # and the score separates OOD from in-distribution with very high AUROC
    s_in = bank.score(held)
    s_ood = np.r_[bank.score(lat), bank.score(nm)]
    r = pd.Series(np.r_[s_ood, s_in]).rank().to_numpy()
    n1 = len(s_ood)
    auroc = (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(s_in))
    assert auroc > 0.99
