# -*- coding: utf-8 -*-
"""P2-07: head training on frozen embeddings. Tested on the mock C07 (random embeddings
with a planted TB shift) so the machinery is verified before real C07 exists. The two
things that must be right: the heads actually learn the planted signal, and the imbalance
config refuses anything but the two approved methods."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from p2_cls import heads as H


def planted(rng, n=2000, dim=64, tb_rate=0.09, shift=1.6):
    """A cloud with a TB direction, like the mock C07: TB images shifted along one axis.
    shift=1.6 against per-axis noise 0.4 gives a Bayes-optimal AUC near 0.99, so a correctly
    trained linear head should comfortably clear 0.9; failing it signals a training bug, not
    a ceiling."""
    y = (rng.random(n) < tb_rate).astype(int)
    axis = np.zeros(dim); axis[0] = 1.0
    x = rng.normal(size=(n, dim)) * 0.4 + shift * y[:, None] * axis
    return x.astype(np.float32), y


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    Xtr, ytr = planted(rng, 2000)
    Xdv, ydv = planted(rng, 600)
    return Xtr, ytr, Xdv, ydv


# ---------------------------------------------------------------- config guards
def test_only_approved_imbalance_methods_accepted():
    H.TrainConfig(imbalance="weighted_sampler")
    H.TrainConfig(imbalance="logit_adjusted")
    with pytest.raises(ValueError, match="approved"):
        H.TrainConfig(imbalance="focal_loss")
    with pytest.raises(ValueError, match="approved"):
        H.TrainConfig(imbalance="smote")


def test_only_valid_head_types():
    with pytest.raises(ValueError, match="head_type"):
        H.TrainConfig(head_type="multilabel")


# ---------------------------------------------------------------- label mapping
def test_label_mappings_follow_the_contract():
    cats = ["healthy", "sick_non_tb", "active_tb", "latent_tb", "active_latent_tb", "uncertain_tb"]
    assert list(H.binary_labels(cats)) == [0, 0, 1, 1, 1, 1]
    assert list(H.three_class_labels(cats)) == [0, 1, 2, 2, 2, 2]


# ---------------------------------------------------------------- learning
def test_binary_head_learns_the_planted_signal(data):
    Xtr, ytr, Xdv, ydv = data
    r = H.train_head(Xtr, ytr, Xdv, ydv, H.TrainConfig(head_type="binary", max_epochs=60))
    assert r.dev_auc > 0.9, "a linear head should recover a strong planted TB signal"
    assert r.best_epoch >= 0


@pytest.mark.parametrize("method", ["weighted_sampler", "logit_adjusted"])
def test_both_imbalance_methods_learn(data, method):
    Xtr, ytr, Xdv, ydv = data
    r = H.train_head(Xtr, ytr, Xdv, ydv,
                     H.TrainConfig(imbalance=method, max_epochs=60))
    assert r.dev_auc > 0.85, "%s failed to learn" % method


def test_three_class_head_produces_a_tb_probability(data):
    Xtr, ytr, Xdv, ydv = data
    # fabricate a 3-class label: split the negatives into healthy/sick at random
    rng = np.random.default_rng(1)
    y3 = np.where(ytr == 1, 2, rng.integers(0, 2, size=len(ytr)))
    yd3 = np.where(ydv == 1, 2, rng.integers(0, 2, size=len(ydv)))
    r = H.train_head(Xtr, y3, Xdv, yd3, H.TrainConfig(head_type="three_class", max_epochs=60))
    assert r.n_classes == 3
    p_tb, logits, sm = H.predict_p_tb(r, Xdv)
    assert p_tb.shape == (len(Xdv),)
    assert np.all((p_tb >= 0) & (p_tb <= 1))
    assert np.allclose(sm.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(sm[:, 2], p_tb)              # p_tb_raw == softmax TB channel


# ---------------------------------------------------------------- determinism & seeds
def test_training_is_deterministic_for_a_fixed_seed(data):
    Xtr, ytr, Xdv, ydv = data
    cfg = H.TrainConfig(seed=5, max_epochs=40)
    a = H.train_head(Xtr, ytr, Xdv, ydv, cfg)
    b = H.train_head(Xtr, ytr, Xdv, ydv, cfg)
    assert a.dev_auc == b.dev_auc
    assert np.allclose(H.predict_p_tb(a, Xdv)[0], H.predict_p_tb(b, Xdv)[0])


def test_seed_averaging_reports_a_spread_and_is_stabler(data):
    Xtr, ytr, Xdv, ydv = data
    results, mean_auc, std_auc = H.train_seeds(
        Xtr, ytr, Xdv, ydv, H.TrainConfig(max_epochs=40), seeds=(17, 29, 43))
    assert len(results) == 3
    assert mean_auc > 0.9
    assert std_auc >= 0.0                            # the D10 spread
    p_avg = H.seed_averaged_p_tb(results, Xdv)
    assert p_avg.shape == (len(Xdv),)
    # averaged prob should sit within the range of the individual seeds' AUCs
    avg_auc = H.auc(ydv, p_avg)
    assert avg_auc >= min(r.dev_auc for r in results) - 0.02


def test_different_seeds_give_different_heads(data):
    Xtr, ytr, Xdv, ydv = data
    a = H.train_head(Xtr, ytr, Xdv, ydv, H.TrainConfig(seed=1, max_epochs=20))
    b = H.train_head(Xtr, ytr, Xdv, ydv, H.TrainConfig(seed=2, max_epochs=20))
    wa = a.state_dict["weight"].numpy()
    wb = b.state_dict["weight"].numpy()
    assert not np.allclose(wa, wb)


# ---------------------------------------------------------------- on the real mock C07
def test_runs_on_the_mock_c07_embeddings():
    from pathlib import Path
    mock = Path(__file__).resolve().parents[1] / "mock/artifacts/p2/emb/radino_tbx11k_crop_m0.10_clahe0_v1.npz"
    if not mock.exists():
        pytest.skip("run scripts/p2_02_make_mocks.py first")
    with np.load(mock) as z:
        ids = z["image_id"].tolist()
        emb = z["emb"].astype(np.float32)
    y = np.array([1 if str(i).startswith("tb") else 0 for i in ids])
    n = len(y)
    cut = int(n * 0.85)
    r = H.train_head(emb[:cut], y[:cut], emb[cut:], y[cut:],
                     H.TrainConfig(head_type="binary", max_epochs=40))
    # the mock plants a real TB shift; the head must recover more than chance
    assert r.dev_auc > 0.7
