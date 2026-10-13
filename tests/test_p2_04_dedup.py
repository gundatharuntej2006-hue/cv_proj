# -*- coding: utf-8 -*-
"""P2-04: the near-duplicate search that gates the Week 2 split freeze (H1).

No model needed, so these run in CI. They are the tests that matter most on this task:
a missed duplicate silently leaks held-out data into training and nothing downstream
ever reveals it.
"""
import math

import numpy as np
import pytest

from p2_gate import dedup as D


def unit(rng, n=8, dim=64):
    x = rng.normal(size=(n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def at_cosine(a, target, rng):
    """A unit vector whose cosine to `a` is exactly `target`."""
    v = rng.normal(size=a.shape)
    v -= a * float(v @ a)
    v /= np.linalg.norm(v)
    b = a + v * math.sqrt(1.0 / target ** 2 - 1.0)
    return b / np.linalg.norm(b)


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_finds_a_duplicate_that_crosses_a_split(rng):
    emb = unit(rng, 6)
    emb[4] = at_cosine(emb[0], 0.995, rng)
    ids = ["a0", "a1", "a2", "a3", "a4", "a5"]
    spl = ["train", "train", "train", "development", "internal_test", "calibration"]
    df, within = D.find_near_duplicates(ids, spl, emb, 0.98)
    assert len(df) == 1 and within == 0
    r = df.iloc[0]
    assert (r.image_id_a, r.image_id_b) == ("a0", "a4")
    assert (r.split_a, r.split_b) == ("train", "internal_test")
    assert r.cosine == pytest.approx(0.995, abs=1e-3)


def test_column_a_always_holds_the_earlier_split(rng):
    """P1 removes the later copy, so getting this backwards deletes the training image
    and keeps the leak."""
    emb = unit(rng, 4)
    emb[0] = at_cosine(emb[3], 0.99, rng)          # index order is the reverse of split order
    ids, spl = ["z0", "z1", "z2", "z3"], ["internal_test", "train", "train", "train"]
    df, _ = D.find_near_duplicates(ids, spl, emb, 0.98)
    assert len(df) == 1
    assert df.iloc[0].split_a == "train" and df.iloc[0].split_b == "internal_test"
    assert df.iloc[0].image_id_a == "z3" and df.iloc[0].image_id_b == "z0"


def test_the_boundary_is_inclusive_and_the_flag_actually_switches_it(rng):
    emb = unit(rng, 2)
    emb[1] = at_cosine(emb[0], 0.98, rng)
    ids, spl = ["p", "q"], ["train", "internal_test"]
    assert len(D.find_near_duplicates(ids, spl, emb, 0.98, inclusive=True)[0]) == 1
    assert len(D.find_near_duplicates(ids, spl, emb, 0.98, inclusive=False)[0]) == 0


def test_a_pair_just_below_tau_is_not_a_candidate(rng):
    emb = unit(rng, 2)
    emb[1] = at_cosine(emb[0], 0.970, rng)
    assert len(D.find_near_duplicates(["p", "q"], ["train", "calibration"], emb, 0.98)[0]) == 0


def test_within_split_pairs_are_counted_but_kept_out_of_c17(rng):
    emb = unit(rng, 4)
    emb[1] = at_cosine(emb[0], 0.999, rng)         # both in train
    df, within = D.find_near_duplicates(["a", "b", "c", "d"],
                                        ["train", "train", "development", "calibration"],
                                        emb, 0.98)
    assert within == 1 and len(df) == 0


def test_blocking_is_an_optimisation_not_a_behaviour_change(rng):
    emb = unit(rng, 40, dim=32)
    for i, j, c in [(1, 30, 0.995), (5, 22, 0.985), (9, 39, 0.9999)]:
        emb[j] = at_cosine(emb[i], c, rng)
    spl = ["train"] * 20 + ["calibration"] * 10 + ["internal_test"] * 10
    ids = ["i%02d" % i for i in range(40)]
    base, _ = D.find_near_duplicates(ids, spl, emb, 0.98, block=1000)
    for b in (1, 3, 7, 16):
        other, _ = D.find_near_duplicates(ids, spl, emb, 0.98, block=b)
        assert other.equals(base), "block size %d changed the result" % b


def test_columns_match_the_contract(rng):
    emb = unit(rng, 3)
    emb[2] = at_cosine(emb[0], 0.99, rng)
    df, _ = D.find_near_duplicates(["a", "b", "c"], ["train", "train", "internal_test"],
                                   emb, 0.98)
    assert list(df.columns) == D.C17_COLUMNS
    assert (df.tau == 0.98).all()


def test_unnormalised_embeddings_are_refused(rng):
    emb = unit(rng, 3) * 2.5
    with pytest.raises(ValueError, match="L2-normalised"):
        D.find_near_duplicates(["a", "b", "c"], ["train"] * 3, emb, 0.98)


def test_unknown_split_names_are_refused(rng):
    with pytest.raises(ValueError, match="unknown split"):
        D.find_near_duplicates(["a", "b"], ["train", "val"], unit(rng, 2), 0.98)


def test_sweep_is_monotone_and_matches_the_search(rng):
    emb = unit(rng, 30, dim=32)
    for i, j, c in [(0, 20, 0.999), (1, 21, 0.985), (2, 22, 0.962), (3, 23, 0.951)]:
        emb[j] = at_cosine(emb[i], c, rng)
    ids = ["s%02d" % i for i in range(30)]
    spl = ["train"] * 20 + ["internal_test"] * 10
    sweep = D.threshold_sweep(ids, spl, emb)
    counts = sweep.cross_split_candidates.tolist()
    assert counts == sorted(counts, reverse=True), "a higher tau cannot find more pairs"
    for t, expected in zip(sweep.tau, counts):
        got, _ = D.find_near_duplicates(ids, spl, emb, float(t))
        assert len(got) == expected, "sweep disagrees with the search at tau=%.2f" % t


def test_load_c07_refuses_a_crop_variant(tmp_path, rng):
    """Dedup on a lung crop would throw away exactly the borders and markers that tell
    two scans of one patient apart from one file stored twice."""
    p = tmp_path / "crop.npz"
    with open(p, "wb") as fh:
        np.savez(fh, image_id=np.array(["a", "b"], dtype="<U32"),
                 emb=unit(rng, 2).astype(np.float16), l2_normalised=np.array(True),
                 variant=np.array("crop_m0.10_clahe0"))
    with pytest.raises(ValueError, match="FULL-image"):
        D.load_c07(p)


def test_load_c07_refuses_a_file_that_is_not_c07(tmp_path, rng):
    p = tmp_path / "junk.npz"
    with open(p, "wb") as fh:
        np.savez(fh, emb=unit(rng, 2))
    with pytest.raises(ValueError, match="not a C07"):
        D.load_c07(p)


def test_load_c07_accepts_the_real_mock(rng):
    from pathlib import Path
    mock = Path(__file__).resolve().parents[1] / "mock/artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz"
    if not mock.exists():
        pytest.skip("run scripts/p2_02_make_mocks.py first")
    ids, emb, variant = D.load_c07(mock)
    assert variant == "full_clahe0" and len(ids) == emb.shape[0]


def test_a_custom_split_order_can_be_supplied(rng):
    """Before C01 exists, the provisional pass runs over TBX11K's own train/val/test,
    which are not C01 split names."""
    emb = unit(rng, 3)
    emb[2] = at_cosine(emb[0], 0.99, rng)
    with pytest.raises(ValueError, match="unknown split"):
        D.find_near_duplicates(["a", "b", "c"], ["train", "train", "val"], emb, 0.98)
    df, _ = D.find_near_duplicates(["a", "b", "c"], ["train", "train", "val"], emb, 0.98,
                                   order=D.OFFICIAL_ORDER)
    assert len(df) == 1
    assert df.iloc[0].split_a == "train" and df.iloc[0].split_b == "val"
