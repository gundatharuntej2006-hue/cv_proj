# -*- coding: utf-8 -*-
"""P2-03 acceptance: deterministic, resumable, contract-valid embeddings.

The model-free tests run everywhere, including CI with no weights. The ones that
actually embed are marked `slow` and skip when the RAD-DINO cache is cold.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_cls import contracts as K
from p2_cls import embed as E


def have_weights():
    home = os.environ.get("HF_HOME")
    return bool(home) and (Path(home) / "hub" / "models--microsoft--rad-dino").exists()


needs_model = pytest.mark.skipif(not have_weights(), reason="RAD-DINO weights not cached")


@pytest.fixture
def images(tmp_path):
    """Twelve distinguishable 512x512 greyscale PNGs."""
    from PIL import Image
    d = tmp_path / "raw"
    d.mkdir()
    rng = np.random.default_rng(7)
    for i in range(12):
        a = rng.integers(30, 220, size=(512, 512), dtype=np.uint8)
        a[: 40 * (i + 1), :] = 255                      # a per-image landmark
        Image.fromarray(a).save(d / ("img%03d.png" % i))
    return d


# ------------------------------------------------------------------ variant parsing
@pytest.mark.parametrize("s,kind,margin,clahe", [
    ("full_clahe0", "full", None, False),
    ("full_clahe1", "full", None, True),
    ("crop_m0.10_clahe0", "crop", 0.10, False),
    ("crop_m0.05_clahe1", "crop", 0.05, True),
])
def test_parse_variant_accepts_the_contract_names(s, kind, margin, clahe):
    v = E.parse_variant(s)
    assert (v.kind, v.margin, v.clahe) == (kind, margin, clahe)
    assert v.name == s, "the parsed variant must round-trip to the same string"


@pytest.mark.parametrize("bad", ["full", "crop_clahe0", "crop_m0.1_clahe0", "full_clahe2",
                                 "FULL_CLAHE0", "crop_m0.10", "full_clahe0 "])
def test_parse_variant_rejects_near_misses(bad):
    """A typo must fail loudly: it would otherwise produce a file that looks correct
    and contains the wrong pixels."""
    with pytest.raises(ValueError):
        E.parse_variant(bad)


# ------------------------------------------------------------------ input resolution
def test_raw_dir_refuses_variants_it_cannot_honestly_produce(images):
    with pytest.raises(SystemExit, match="CLAHE"):
        E.load_manifest(None, E.parse_variant("full_clahe1"), raw_dir=images)
    with pytest.raises(SystemExit, match="crop"):
        E.load_manifest(None, E.parse_variant("crop_m0.10_clahe0"), raw_dir=images)


def test_raw_dir_is_stamped_as_a_standin(images):
    df, preproc = E.load_manifest(None, E.parse_variant("full_clahe0"), raw_dir=images)
    assert preproc == E.STANDIN_PREPROC != E.REAL_PREPROC
    assert len(df) == 12 and df.crop_box.isna().all()


def test_manifest_paths_resolve_relative_to_the_manifest(tmp_path, images):
    man = tmp_path / "manifest.csv"
    pd.DataFrame({"image_id": ["img000"], "path_img512": ["raw/img000.png"]}).to_csv(man, index=False)
    df, preproc = E.load_manifest(man, E.parse_variant("full_clahe0"))
    assert preproc == E.REAL_PREPROC
    assert Path(df.src_path[0]).exists()


def test_crop_variant_demands_c05(tmp_path, images):
    man = tmp_path / "manifest.csv"
    pd.DataFrame({"image_id": ["img000"], "path_img512": ["raw/img000.png"]}).to_csv(man, index=False)
    with pytest.raises(SystemExit, match="--roi"):
        E.load_manifest(man, E.parse_variant("crop_m0.10_clahe0"))


def test_read_image_applies_the_crop_box(images):
    full = E.read_image(images / "img000.png", None)
    crop = E.read_image(images / "img000.png", (10, 20, 110, 220))
    assert full.shape == (512, 512) and crop.shape == (200, 100)
    assert np.array_equal(crop, full[20:220, 10:110])


# ------------------------------------------------------------------ the C07 output
@needs_model
def test_output_satisfies_the_c07_contract(tmp_path, images):
    v = E.parse_variant("full_clahe0")
    df, preproc = E.load_manifest(None, v, raw_dir=images)
    out, n = E.run(df, v, tmp_path / "emb.npz", preproc, batch=4, log=lambda *_: None)
    z = np.load(out)
    assert sorted(z.files) == sorted(K.C07_KEYS)
    assert z["emb"].shape == (12, K.EMB_DIM) and z["emb"].dtype == np.float16
    assert bool(z["l2_normalised"]) is True
    assert str(z["variant"]) == "full_clahe0"
    assert E.REVISION in str(z["model_id"])
    assert str(z["preproc_version"]) == E.STANDIN_PREPROC
    norms = np.linalg.norm(z["emb"].astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=2e-3)
    assert list(z["image_id"]) == sorted(df.image_id)        # deterministic ordering
    assert not Path(str(out) + ".partial").exists(), "shards must be cleaned up"


@needs_model
def test_rerun_is_identical(tmp_path, images):
    """The acceptance criterion is cosine >= 0.9999; fixed batching should actually
    give bit-equality, so assert the stronger thing and learn if it ever weakens."""
    v = E.parse_variant("full_clahe0")
    df, preproc = E.load_manifest(None, v, raw_dir=images)
    a, _ = E.run(df, v, tmp_path / "a.npz", preproc, batch=4, log=lambda *_: None)
    b, _ = E.run(df, v, tmp_path / "b.npz", preproc, batch=4, log=lambda *_: None)
    ea, eb = np.load(a)["emb"], np.load(b)["emb"]
    cos = (ea.astype(np.float32) * eb.astype(np.float32)).sum(1)
    assert cos.min() >= 0.9999
    assert np.array_equal(ea, eb)


@needs_model
def test_resume_after_a_crash_reproduces_the_uninterrupted_result(tmp_path, images, monkeypatch):
    v = E.parse_variant("full_clahe0")
    df, preproc = E.load_manifest(None, v, raw_dir=images)
    monkeypatch.setattr(E, "SHARD", 4)                       # 12 images -> 3 shards

    ref, _ = E.run(df, v, tmp_path / "ref.npz", preproc, batch=2, log=lambda *_: None)
    reference = np.load(ref)["emb"]

    real_write = E.write_shard
    def explode(out, idx, ids, emb):
        real_write(out, idx, ids, emb)
        if idx == 1:
            raise RuntimeError("simulated crash after shard 1")
    monkeypatch.setattr(E, "write_shard", explode)
    with pytest.raises(RuntimeError):
        E.run(df, v, tmp_path / "x.npz", preproc, batch=2, log=lambda *_: None)
    partial = Path(str(tmp_path / "x.npz") + ".partial")
    assert sorted(p.name for p in partial.glob("*.npz")) == ["shard_00000.npz", "shard_00001.npz"]
    assert not list(partial.glob("*.tmp")), "no half-written shard may survive"

    monkeypatch.setattr(E, "write_shard", real_write)
    out, n = E.run(df, v, tmp_path / "x.npz", preproc, batch=2, log=lambda *_: None)
    assert n == 12
    assert np.array_equal(np.load(out)["emb"], reference)


@needs_model
def test_crop_variant_embeds_the_crop_not_the_full_image(tmp_path, images):
    man = tmp_path / "manifest.csv"
    ids = ["img%03d" % i for i in range(4)]
    pd.DataFrame({"image_id": ids,
                  "path_img512": ["raw/%s.png" % i for i in ids]}).to_csv(man, index=False)
    roi = tmp_path / "roi.parquet"
    pd.DataFrame({"image_id": ids, "margin": [0.10] * 4,
                  "crop_x0": [40] * 4, "crop_y0": [60] * 4,
                  "crop_x1": [470] * 4, "crop_y1": [450] * 4}).to_parquet(roi)
    vf, vc = E.parse_variant("full_clahe0"), E.parse_variant("crop_m0.10_clahe0")
    df_f, pv = E.load_manifest(man, vf)
    df_c, _ = E.load_manifest(man, vc, roi_path=roi)
    assert df_c.crop_box.tolist() == [(40, 60, 470, 450)] * 4
    f, _ = E.run(df_f, vf, tmp_path / "f.npz", pv, batch=4, log=lambda *_: None)
    c, _ = E.run(df_c, vc, tmp_path / "c.npz", pv, batch=4, log=lambda *_: None)
    ef, ec = np.load(f)["emb"].astype(np.float32), np.load(c)["emb"].astype(np.float32)
    assert (ef * ec).sum(1).max() < 0.999, "cropping changed nothing; is the box applied?"
    assert str(np.load(c)["variant"]) == "crop_m0.10_clahe0"


@needs_model
def test_verify_reports_the_acceptance_criterion(tmp_path, images):
    v = E.parse_variant("full_clahe0")
    df, preproc = E.load_manifest(None, v, raw_dir=images)
    out, _ = E.run(df, v, tmp_path / "e.npz", preproc, batch=4, log=lambda *_: None)
    worst, checked = E.verify(out, df, v, sample=6, batch=4, log=lambda *_: None)
    assert checked == 6 and worst >= 0.9999
