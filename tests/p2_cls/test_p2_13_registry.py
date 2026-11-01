# -*- coding: utf-8 -*-
"""P2-13: model registry + the R2 reproduction guarantee. The acceptance criterion is that
re-running a registered head reproduces its exported C08 scores to 1e-4; the test trains a
head, exports scores, registers it, and checks reproduction end to end."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from p2_cls import heads as H
from p2_cls import registry as R


def planted(rng, n=1500, dim=48, tb_rate=0.1, shift=1.6):
    y = (rng.random(n) < tb_rate).astype(int)
    axis = np.zeros(dim); axis[0] = 1.0
    x = (rng.normal(size=(n, dim)) * 0.4 + shift * y[:, None] * axis).astype(np.float32)
    return x, y


@pytest.fixture
def trained(tmp_path):
    rng = np.random.default_rng(0)
    Xtr, ytr = planted(rng, 1500)
    Xdv, ydv = planted(rng, 400)
    ids = ["img%04d" % i for i in range(len(Xdv))]
    res = H.train_head(Xtr, ytr, Xdv, ydv, H.TrainConfig(head_type="binary", max_epochs=40))
    wp = tmp_path / "head.npz"
    sha = R.save_head(res, wp)
    entry = R.RegistryEntry(
        model_id="radino_crop_m0.10_clahe0_bin", head_type="binary",
        variant="crop_m0.10_clahe0", backbone="microsoft/rad-dino@110cbc18",
        weight_path=str(wp), weight_sha256=sha, n_classes=2, is_primary=True)
    return entry, res, Xdv, ydv, ids


def test_saved_head_reproduces_the_training_forward_pass(trained):
    entry, res, Xdv, ydv, ids = trained
    # the registry's numpy forward must match the torch head used in training
    p_torch, _, _ = H.predict_p_tb(res, Xdv)
    p_numpy = R.classify_embeddings(entry, Xdv)
    assert np.max(np.abs(p_torch - p_numpy)) < 1e-5


def test_r2_reproduces_exported_c08_to_1e4(trained):
    """The headline P2-13 acceptance criterion."""
    entry, res, Xdv, ydv, ids = trained
    # export C08-style scores from the same head
    p = R.classify_embeddings(entry, Xdv)
    c08 = pd.DataFrame({"image_id": ids, "model_id": entry.model_id, "p_tb_raw": p})
    ok, maxdiff, n = R.reproduces_scores(entry, Xdv, ids, c08, tol=1e-4)
    assert ok and n == len(ids)
    assert maxdiff <= 1e-4


def test_reproduction_fails_if_weights_are_tampered(trained, tmp_path):
    entry, res, Xdv, ydv, ids = trained
    p = R.classify_embeddings(entry, Xdv)
    c08 = pd.DataFrame({"image_id": ids, "model_id": entry.model_id, "p_tb_raw": p})
    # corrupt the saved head
    W, b, n = R.load_head(entry.weight_path)
    with open(entry.weight_path, "wb") as fh:
        np.savez(fh, W=W * 1.5, b=b, n_classes=np.array(n))
    ok, maxdiff, _ = R.reproduces_scores(entry, Xdv, ids, c08, tol=1e-4)
    assert not ok and maxdiff > 1e-4


# ---------------------------------------------------------------- registry integrity
def test_registry_enforces_single_primary(trained, tmp_path):
    entry, *_ = trained
    reg = R.Registry()
    reg.add(entry)
    dup = R.RegistryEntry(**{**entry.to_json(), "model_id": "radino_full_clahe0_bin",
                             "is_primary": True})
    with pytest.raises(ValueError, match="primary"):
        reg.add(dup)


def test_registry_rejects_duplicate_model_id(trained):
    entry, *_ = trained
    reg = R.Registry([entry])
    with pytest.raises(ValueError, match="duplicate"):
        reg.add(R.RegistryEntry(**{**entry.to_json(), "is_primary": False}))


def test_registry_round_trips_and_verifies_shas(trained, tmp_path):
    entry, *_ = trained
    reg = R.Registry([entry])
    path = tmp_path / "registry.json"
    reg.save(path)
    reg2 = R.Registry.load(path)
    assert reg2.primary().model_id == entry.model_id
    assert reg2.verify_shas() == []            # weight file intact


def test_sha_mismatch_is_detected(trained):
    entry, *_ = trained
    reg = R.Registry([entry])
    W, b, n = R.load_head(entry.weight_path)
    with open(entry.weight_path, "wb") as fh:
        np.savez(fh, W=W + 1, b=b, n_classes=np.array(n))
    bad = reg.verify_shas()
    assert bad and bad[0][1] == "SHA mismatch"


def test_the_registry_covers_the_five_methodology_models():
    """The registry must be able to hold exactly the comparison set P2-13 ships."""
    from p2_cls import contracts as K
    ids = {m["model_id"] for m in K.REGISTRY}
    assert sum(m["is_primary"] for m in K.REGISTRY) == 1
    assert any(m["head"] == "three_class" for m in K.REGISTRY)      # alternate head
    assert any(m["margin"] is None for m in K.REGISTRY)             # full-image variant
    assert any(m["clahe"] for m in K.REGISTRY)                      # CLAHE alternate
    assert any(i.startswith("effb3") for i in ids)                  # EfficientNet baseline
