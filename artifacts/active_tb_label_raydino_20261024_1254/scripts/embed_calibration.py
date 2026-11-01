# -*- coding: utf-8 -*-
"""Extend the frozen regional feature cache to the 899 TBX11K calibration images.

The existing cache (`artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz`) covers only
train + development (6,600 images, the official TBX11K train list). Calibration and
internal_test come from the official val list and were never regionally embedded.

This embeds **calibration only**. `internal_test` is excluded by an assertion.

Nothing in `artifacts/p2/` or `data/processed/` is written. The lungs-masked PNGs are
rebuilt into this run's own scratch directory and the npz lands in this run directory.

Before doing anything it re-embeds 24 *development* images that are already in the cache
and asserts they reproduce, which verifies the checkpoint, the revision, the preprocessing
and the pooling grid rather than assuming them.

    py -3 embed_calibration.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

OUT = RUN / "outputs" / "regional_tbx11k_calibration_lungs_g12_v1.npz"
SCRATCH = RUN / "outputs" / "lungs512_calibration"
CACHE = ROOT / "artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz"
C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
SRC = ROOT / "data/processed/v1/tbx11k/img512"
MASKS = ROOT / "data/processed/v1/masks"
GRID = 12
TOL = 2e-3          # float16 storage, so exact equality is not the right test


def lungs_png(stem, out_dir):
    from PIL import Image
    dst = out_dir / (stem + ".png")
    if dst.exists():
        return dst
    a = np.array(Image.open(SRC / (stem + ".png")).convert("L"), np.uint8)
    mp = MASKS / (stem + ".png")
    if mp.exists():
        m = np.array(Image.open(mp).convert("L"), np.uint8) > 127
        if m.sum():
            o = a.astype(np.float32)
            o[~m] = float(o.mean())
            a = np.clip(o, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(dst)
    return dst


def embed_paths(embedder, paths, batch=16):
    from PIL import Image
    ids, vecs, buf, bids = [], [], [], []
    for p in paths:
        buf.append(np.array(Image.open(p).convert("L"), dtype=np.uint8))
        bids.append(p.stem)
        if len(buf) >= batch:
            vecs.append(embedder.embed(buf))
            ids.extend(bids)
            buf, bids = [], []
    if buf:
        vecs.append(embedder.embed(buf))
        ids.extend(bids)
    return ids, np.concatenate(vecs)


def main():
    import pandas as pd
    import p2_31_regional_mil as R

    assert R.GRID == GRID, "p2_31 GRID is %d, expected %d" % (R.GRID, GRID)
    c01 = pd.read_csv(C01)
    cal = sorted(c01[c01.split == "calibration"].image_id)
    forbidden = set(c01[c01.split == "internal_test"].image_id)
    assert len(cal) == 899, "expected 899 calibration images, got %d" % len(cal)
    assert not (set(cal) & forbidden), "internal_test leaked into the embedding set"

    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    embedder = R.RegionalEmbedder()
    report = {"backbone": "microsoft/rad-dino",
              "revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
              "patch_grid": int(embedder.g), "pooled_grid": GRID,
              "device": embedder.dev, "half_precision": bool(embedder.half)}

    # --- verification: reproduce 24 cached development vectors --------------------------
    with np.load(CACHE) as z:
        cid, CX = z["image_id"].tolist(), z["reg"]
    assert CX.shape[1:] == (GRID * GRID, 768), "cache shape %s unexpected" % (CX.shape,)
    ci = {i: k for k, i in enumerate(cid)}
    dev = [i for i in sorted(c01[c01.split == "development"].image_id) if i in ci][:24]
    vpaths = [lungs_png(s, SCRATCH) for s in dev]
    t0 = time.perf_counter()
    vids, VV = embed_paths(embedder, vpaths)
    ref = CX[[ci[i] for i in vids]].astype(np.float32)
    got = VV.astype(np.float32)
    mad = float(np.abs(ref - got).mean())
    mx = float(np.abs(ref - got).max())
    cos = float(np.mean(np.sum(ref * got, axis=2)
                        / (np.linalg.norm(ref, axis=2) * np.linalg.norm(got, axis=2) + 1e-12)))
    report["verification"] = {"n_images": len(vids), "mean_abs_diff": round(mad, 6),
                              "max_abs_diff": round(mx, 6),
                              "mean_cosine_to_cache": round(cos, 6),
                              "tolerance": TOL,
                              "shape_match": list(VV.shape[1:]) == [GRID * GRID, 768]}
    print("verification on %d cached development images: MAD %.2e  max %.2e  cos %.6f"
          % (len(vids), mad, mx, cos))
    if mx > TOL or cos < 0.9999:
        for p in vpaths:
            p.unlink(missing_ok=True)
        (RUN / "outputs" / "embed_calibration.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
        raise SystemExit("re-embedding does NOT reproduce the cache; refusing to extend it")
    print("  reproduces the frozen cache within tolerance")
    for p in vpaths:
        p.unlink(missing_ok=True)

    # --- the calibration set --------------------------------------------------------------
    paths = [lungs_png(s, SCRATCH) for s in cal]
    ids, arr = embed_paths(embedder, paths)
    assert len(ids) == 899 and arr.shape == (899, GRID * GRID, 768), \
        "unexpected output shape %s for %d ids" % (arr.shape, len(ids))
    assert not (set(ids) & forbidden)
    np.savez_compressed(OUT, image_id=np.asarray(ids, dtype="<U48"), reg=arr)
    for p in paths:
        p.unlink(missing_ok=True)
    SCRATCH.rmdir()

    report.update({"n_calibration": len(ids), "shape": list(arr.shape),
                   "dtype": str(arr.dtype), "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
                   "output": str(OUT.relative_to(ROOT)),
                   "internal_test_embedded": False})
    (RUN / "outputs" / "embed_calibration.json").write_text(json.dumps(report, indent=1),
                                                            encoding="utf-8")
    print("wrote %s  %s  (%.1f min)" % (OUT.name, arr.shape, (time.perf_counter() - t0) / 60))


if __name__ == "__main__":
    main()
