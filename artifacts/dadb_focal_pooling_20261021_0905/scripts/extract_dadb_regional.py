# -*- coding: utf-8 -*-
"""Phase 1a: extract frozen RAD-DINO regional (12x12) features for DA/DB.

No model is trained here and no label is read. The preprocessing rule is copied from the
committed path so DA/DB matches TBX11K, Vietnam and the NLM sets:

  * `scripts/p2_26_dadb_third_site.py::embed_variant(variant="lungs_only")` -- resize to
    512x512 with INTER_AREA if needed, mask with `data/processed/v1/masks_dadb`, fill
    non-lung pixels with the image mean;
  * `scripts/p2_36_cross_dataset_pooling.py::stage_embed` -- feed those PNGs to
    `p2_31_regional_mil.RegionalEmbedder`, which must report a 37x37 patch grid and pool
    to GRID x GRID.

Both the patch grid (37) and the output region count (144) are asserted explicitly. A
silent 18x18-against-12x12 mismatch cost an earlier run a full experiment; it cannot pass
unnoticed here.

Output goes to the run directory, NOT to artifacts/p2/regional/, so no existing cache is
touched.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SRC = ROOT / "data/raw/tbx11k/TBX11K/imgs/extra/da+db"
MASK_DIR = ROOT / "data/processed/v1/masks_dadb"
LUNGS_PNG = RUN / "lungs_png"
OUT = RUN / "tables/regional_dadb_lungs_g12_v1.npz"
EXPECT_GRID = 12
EXPECT_REGIONS = EXPECT_GRID * EXPECT_GRID
EXPECT_PATCH_GRID = 37
EXPECT_N = 176


def cases():
    """Exactly the ordering p2_26::images() produces. No label is used in this module."""
    out = []
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            out.append(p)
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_lungs_png(paths, log=print):
    import cv2
    from PIL import Image
    LUNGS_PNG.mkdir(parents=True, exist_ok=True)
    made, skipped = 0, []
    for p in paths:
        dst = LUNGS_PNG / (p.stem + ".png")
        if dst.exists():
            continue
        mp = MASK_DIR / (p.stem + ".png")
        if not mp.exists():
            skipped.append((p.stem, "no lung mask"))
            continue
        a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        if a.shape != (512, 512):
            a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
        m = np.array(Image.open(mp).convert("L"), np.uint8) > 127
        if m.sum() == 0:
            skipped.append((p.stem, "empty lung mask"))
            continue
        o = a.astype(np.float32)
        o[~m] = float(o.mean())
        Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).save(dst)
        made += 1
    log("  lungs-only PNGs: %d present (%d written this run), %d skipped"
        % (len(list(LUNGS_PNG.glob("*.png"))), made, len(skipped)))
    for s, why in skipped:
        log("    SKIPPED %s: %s" % (s, why))
    return skipped


def main():
    from PIL import Image
    import p2_31_regional_mil as RM

    if RM.GRID != EXPECT_GRID:
        raise SystemExit("p2_31_regional_mil.GRID is %d, expected %d -- refusing to run"
                         % (RM.GRID, EXPECT_GRID))
    print("asserted p2_31_regional_mil.GRID == %d" % RM.GRID)

    paths = cases()
    if len(paths) != EXPECT_N:
        raise SystemExit("found %d DA/DB files, expected %d" % (len(paths), EXPECT_N))
    skipped = build_lungs_png(paths)
    if skipped:
        raise SystemExit("refusing to proceed with %d skipped cases" % len(skipped))

    if OUT.exists():
        print("  cached: %s" % OUT.name)
        return

    emb = RM.RegionalEmbedder(batch=16)
    if emb.g != EXPECT_PATCH_GRID:
        raise SystemExit("unexpected RAD-DINO patch grid %d, expected %d"
                         % (emb.g, EXPECT_PATCH_GRID))
    print("asserted RAD-DINO patch grid == %d" % emb.g)

    order = [p.stem for p in paths]
    ids, vecs, buf, bids = [], [], [], []
    t0 = time.perf_counter()
    for stem in order:
        buf.append(np.array(Image.open(LUNGS_PNG / (stem + ".png")).convert("L"),
                            dtype=np.uint8))
        bids.append(stem)
        if len(buf) >= 16:
            vecs.append(emb.embed(buf))
            ids.extend(bids)
            buf, bids = [], []
    if buf:
        vecs.append(emb.embed(buf))
        ids.extend(bids)
    arr = np.concatenate(vecs)

    if arr.shape != (EXPECT_N, EXPECT_REGIONS, 768):
        raise SystemExit("extracted %s, expected (%d, %d, 768)"
                         % (arr.shape, EXPECT_N, EXPECT_REGIONS))
    if ids != order:
        raise SystemExit("id ordering diverged from the source manifest")
    print("asserted output shape %s and id ordering" % (arr.shape,))

    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U48"), reg=arr)
    os.replace(tmp, OUT)

    meta = {"n": len(ids), "shape": list(arr.shape), "dtype": str(arr.dtype),
            "grid": EXPECT_GRID, "regions": EXPECT_REGIONS,
            "patch_grid": int(emb.g),
            "preprocessing": "resize 512 INTER_AREA if needed; lungs-only mask from "
                             "data/processed/v1/masks_dadb; non-lung filled with image mean",
            "source_order": "sorted(SRC.rglob('*')) over "
                            "data/raw/tbx11k/TBX11K/imgs/extra/da+db",
            "image_id_stored": True,
            "labels_used": False,
            "sha256": sha(OUT),
            "minutes": round((time.perf_counter() - t0) / 60, 2)}
    json.dump(meta, open(RUN / "tables/regional_extraction.json", "w", encoding="utf-8"),
              indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
