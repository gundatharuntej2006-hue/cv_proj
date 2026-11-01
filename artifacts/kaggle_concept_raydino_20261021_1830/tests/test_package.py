# -*- coding: utf-8 -*-
"""Synthetic tests for the Kaggle package. No weights, no real data, no GPU.

    py -3 -m pytest artifacts/kaggle_concept_raydino_20261021_1830/tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))

import guards as G                                                   # noqa: E402
import evaluate as E                                                 # noqa: E402


# =============================================================== locked partition
def test_locked_path_is_refused():
    for p in ("data/internal_test/x.png", "a/INTERNAL_TEST/b", "x/internal-test/y",
              "features_internaltest.npz"):
        with pytest.raises(G.LockedPartitionError):
            G.assert_path_allowed(p)


def test_allowed_paths_pass():
    for p in ("data/train/x.png", "artifacts/p2/development.csv", "vietnam_img512/1.png"):
        assert G.assert_path_allowed(p)


def test_safe_listdir_skips_locked_subtree(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "internal_test").mkdir()
    (tmp_path / "train" / "a.png").write_bytes(b"x")
    (tmp_path / "internal_test" / "b.png").write_bytes(b"x")
    got = [p.name for p in G.safe_listdir(tmp_path, "*")]
    assert "train" in got
    assert not any("internal" in g for g in got)


def test_locked_rows_are_dropped_and_counted():
    df = pd.DataFrame({"image_id": list("abcdef"),
                       "split": ["train", "train", "development", "internal_test",
                                 "internal_test", "calibration"]})
    usable, n = G.drop_locked_rows(df)
    assert n == 2
    assert len(usable) == 4
    assert "internal_test" not in set(usable.split)


def test_payload_guard_rejects_locked_content():
    with pytest.raises(G.LockedPartitionError):
        G.assert_no_locked_in({"rows": [{"id": "x", "split": "internal_test_row_7"}]},
                              "payload")


# =============================================================== shapes
@pytest.mark.parametrize("enc,dim,grid", [("RAD-DINO", 768, 37), ("RayDINO", 1024, 32)])
def test_expected_encoder_shapes(enc, dim, grid):
    n = 4
    glob = np.zeros((n, dim), np.float16)
    reg = np.zeros((n, grid * grid, dim), np.float16)
    assert G.assert_shape(glob, (n, dim), "%s global" % enc)
    assert G.assert_shape(reg, (n, grid * grid, dim), "%s regional" % enc)


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        G.assert_shape(np.zeros((4, 768)), (4, 1024), "wrong dim")


def test_pool_to_grid_output_shape_and_normalisation():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(PKG / "src"))
    from encoders import pool_to_grid
    x = torch.randn(3, 32 * 32, 1024).half()
    out = pool_to_grid(x, 32, 12)
    assert tuple(out.shape) == (3, 144, 1024)
    n = out.float().norm(dim=2)
    assert float(n.min()) > 0.99 and float(n.max()) < 1.01


def test_pool_to_grid_rejects_wrong_src_grid():
    torch = pytest.importorskip("torch")
    from encoders import pool_to_grid
    x = torch.randn(2, 100, 768).half()
    with pytest.raises(AssertionError):
        pool_to_grid(x, 37, 12)          # 37*37 != 100


# =============================================================== row alignment
def test_id_alignment_detects_reordering():
    a = ["p1", "p2", "p3"]
    assert G.assert_ids_aligned(a, list(a))
    with pytest.raises(AssertionError):
        G.assert_ids_aligned(a, ["p1", "p3", "p2"])


def test_id_alignment_detects_duplicates_and_length():
    with pytest.raises(AssertionError):
        G.assert_ids_aligned(["p1", "p1"], ["p1", "p1"])
    with pytest.raises(AssertionError):
        G.assert_ids_aligned(["p1", "p2"], ["p1"])


# =============================================================== target semantics
def test_unknown_target_mapping_is_rejected():
    with pytest.raises(ValueError):
        G.assert_target_mapping(None)
    with pytest.raises(ValueError):
        G.assert_target_mapping({"positive_categories": ["active_tb"]})


def test_overlapping_target_classes_rejected():
    with pytest.raises(ValueError):
        G.assert_target_mapping({"label_column": "tb_label",
                                 "positive_categories": ["active_tb", "healthy"],
                                 "negative_categories": ["healthy"],
                                 "description": "bad"})


def test_reference_target_mapping_is_accepted():
    assert G.assert_target_mapping({
        "label_column": "tb_label",
        "positive_categories": ["active_tb", "latent_tb", "active_latent_tb"],
        "negative_categories": ["healthy"],
        "description": "frozen baseline target; latent_tb is POSITIVE here even though "
                       "active_tb_label is 0 - a documented limitation applied to both arms"})


# =============================================================== thresholds
def test_threshold_refuses_non_development_partition():
    s = np.random.default_rng(0).normal(size=200)
    y = (np.arange(200) % 5 == 0).astype(int)
    assert E.fit_threshold(s, y, "development") is not None
    for bad in ("report", "internal_test", "tune", "calibration"):
        with pytest.raises(ValueError):
            E.fit_threshold(s, y, bad)


def test_threshold_hits_its_target_specificity():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(800), np.ones(200)].astype(int)
    s = np.r_[rng.normal(0, 1, 800), rng.normal(2, 1, 200)]
    t = E.fit_threshold(s, y, "development", 0.70)
    assert abs(E.operating_point(s, y, t)["specificity"] - 0.70) < 0.02


def test_threshold_refuses_too_few_negatives():
    with pytest.raises(ValueError):
        E.fit_threshold(np.zeros(12), np.r_[np.ones(9), np.zeros(3)], "development")


# =============================================================== metrics
def test_auroc_of_a_perfect_and_a_random_score():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    assert E.metrics_block(y, np.r_[np.zeros(50), np.ones(50)], "perfect")["auroc"] == 1.0
    b = E.metrics_block(y, np.random.default_rng(0).normal(size=100), "random")
    assert 0.3 < b["auroc"] < 0.7


def test_metrics_block_handles_a_single_class_group():
    b = E.metrics_block(np.ones(10, int), np.random.rand(10), "all positive")
    assert b["auroc"] is None and "fewer than two classes" in b["note"]


def test_metrics_block_handles_an_empty_group():
    b = E.metrics_block(np.array([], int), np.array([]), "empty")
    assert b["n"] == 0 and b["auroc"] is None


def test_sens_at_spec_and_spec_at_sens_are_consistent():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    s = np.r_[rng.normal(0, 1, 500), rng.normal(3, 1, 500)]
    assert E.sens_at_spec(s, y, 0.70) > 0.8
    assert E.spec_at_sens(s, y, 0.90) > 0.5


def test_paired_bootstrap_detects_a_real_difference_and_a_null():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(400), np.ones(400)].astype(int)
    weak = np.r_[rng.normal(0, 1, 400), rng.normal(0.4, 1, 400)]
    strong = np.r_[rng.normal(0, 1, 400), rng.normal(2.0, 1, 400)]
    d = E.paired_boot(y, weak, strong, n_boot=600, seed=0)
    assert d["delta"] > 0 and d["excludes_zero"]
    same = E.paired_boot(y, weak, weak.copy(), n_boot=600, seed=0)
    assert abs(same["delta"]) < 1e-12 and not same["excludes_zero"]


def test_patient_level_bootstrap_widens_the_interval():
    """Duplicated patients must not be treated as independent cases."""
    rng = np.random.default_rng(4)
    y0 = np.r_[np.zeros(100), np.ones(100)].astype(int)
    s0 = np.r_[rng.normal(0, 1, 100), rng.normal(1, 1, 100)]
    y = np.repeat(y0, 3)
    s = np.repeat(s0, 3)
    pid = np.repeat(np.arange(200), 3)
    lo_c, hi_c = E.boot_ci(y, s, n_boot=600, seed=0)
    lo_p, hi_p = E.boot_ci(y, s, n_boot=600, seed=0, groups=pid)
    assert (hi_p - lo_p) > (hi_c - lo_c)


def test_bh_correction_behaviour():
    adj, rej = E.bh_correct([0.001, 0.02, 0.5, 0.9])
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]
    assert rej[0] and not rej[-1]
    assert np.all((adj >= 0) & (adj <= 1))
    a, r = E.bh_correct([])
    assert len(a) == 0 and len(r) == 0


def test_C_selection_uses_only_the_development_split():
    rng = np.random.default_rng(5)
    xt, yt = rng.normal(size=(300, 8)), (rng.random(300) > 0.7).astype(int)
    xd, yd = rng.normal(size=(120, 8)), (rng.random(120) > 0.7).astype(int)
    out = E.select_C(xt, yt, xd, yd, grid=(0.1, 1.0))
    assert out["C"] in (0.1, 1.0) and "development_auroc" in out


def test_head_refuses_single_class_training_data():
    with pytest.raises(ValueError):
        E.fit_head(np.zeros((10, 4)), np.zeros(10, int))


# =============================================================== package integrity
def test_every_notebook_is_valid_json_and_refuses_locked_data():
    import json
    nbs = sorted((PKG / "kaggle").glob("*.ipynb"))
    assert len(nbs) == 4, [p.name for p in nbs]
    for nb in nbs:
        doc = json.loads(nb.read_text(encoding="utf-8"))
        assert doc["nbformat"] == 4
        src = "\n".join("".join(c["source"]) for c in doc["cells"])
        # a notebook must either read from the Kaggle input mount, or be explicitly
        # gated off before it touches anything (notebook 03 is disabled by default)
        reads_inputs = "/kaggle/input" in src
        gated_off = "ENABLED = False" in src and "SystemExit" in src
        assert reads_inputs or gated_off, (
            "%s neither reads /kaggle/input nor is explicitly disabled" % nb.name)
        # no notebook may construct a locked path
        assert "internal_test/" not in src.replace("'internal_test'", "")


def test_no_patient_data_or_weights_in_the_package():
    bad = []
    for p in PKG.rglob("*"):
        if p.is_file() and p.suffix.lower() in (
                ".png", ".jpg", ".jpeg", ".dcm", ".npz", ".pt", ".pth", ".bin",
                ".safetensors"):
            bad.append(p.relative_to(PKG).as_posix())
    assert not bad, "package must be code-only; found %s" % bad
