"""P3-03: MOCK generator for C11 (Grad-CAM heatmaps + localisation scores). Fake data, valid shape.

    python mock/make_c11.py --split development --n 30
writes mock/artifacts/p3/xai/{model_id}/{split}/{image_id}.npy (float16 [512,512], [0,1], zero outside crop)
   and mock/artifacts/p3/xai/loc_{model_id}_{split}_v1.parquet

Heatmaps are written for every mock image; table rows only for images with GT boxes (TB images),
as decided in P3-10. Scores come from p3_xai.metrics on the written heatmaps, so they are consistent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import _p3_common as c  # noqa: E402
from p3_xai.metrics import localisation_row  # noqa: E402

LOC_SCHEMA = pa.schema([
    ("image_id", pa.string()), ("model_id", pa.string()),
    ("peak_x", pa.int16()), ("peak_y", pa.int16()),
    ("pointing_hit", pa.bool_()), ("energy_in_box", pa.float32()),
    ("n_gt_boxes", pa.int16()), ("tb_subtype", pa.string()),
])
SUBTYPES = ("active", "latent", "active+latent")


def fake_heatmap(image_id: str, seed: int = 0) -> np.ndarray:
    crop = c.fake_crop_box(image_id, seed)
    x0, y0, x1, y1 = crop
    r = c.rng_for(image_id, seed, salt=4)
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in c.fake_gt_boxes(image_id, crop, seed)]
    centres += [tuple(r.uniform([x0, y0], [x1, y1])) for _ in range(2)]
    h = np.zeros((512, 512), np.float32)
    for k, (cx, cy) in enumerate(centres):
        s = float(r.uniform(15, 45))
        h += (1.0 if k == 0 else 0.5) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    mask = np.zeros_like(h, bool)
    mask[y0:y1, x0:x1] = True
    h[~mask] = 0.0
    h /= h.max()
    return h.astype(np.float16)


def generate(split: str = "development", n: int = 30, seed: int = 0, model_id: str = c.CAM_MODEL_ID,
             root: Path | str | None = None) -> Path:
    rows = []
    for image_id in c.mock_image_ids(n, seed):
        hm = fake_heatmap(image_id, seed)
        np.save(c.out_path(root, "artifacts", "p3", "xai", model_id, split, f"{image_id}.npy"), hm)
        gt = c.fake_gt_boxes(image_id, c.fake_crop_box(image_id, seed), seed)
        if gt:
            subtype = SUBTYPES[int(c.rng_for(image_id, seed, 5).integers(0, 3))]
            rows.append(localisation_row(image_id, model_id, hm, gt, subtype).__dict__)
    path = c.out_path(root, "artifacts", "p3", "xai", f"loc_{model_id}_{split}_v1.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=LOC_SCHEMA), path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="development")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-id", default=c.CAM_MODEL_ID)
    ap.add_argument("--root", default=None, help="output root (default: mock/)")
    a = ap.parse_args(argv)
    print(generate(a.split, a.n, a.seed, a.model_id, a.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
