"""P3-03: MOCK generator for C10 (detections). Fake data, valid shape.

    python mock/make_c10.py --dataset tbx11k --split development --n 30
writes mock/artifacts/p3/det/detections_{dataset}_{split}_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _p3_common as c  # noqa: E402

MIN_SCORE = 0.01          # C10: every box with score >= 0.01 is kept
CONF_THRESHOLD = 0.42     # stands in for D18
INPUT_SIZE = 512          # stands in for D16


def fake_image_entry(image_id: str, seed: int = 0, conf_threshold: float = CONF_THRESHOLD) -> dict:
    crop = c.fake_crop_box(image_id, seed)
    r = c.rng_for(image_id, seed, salt=3)
    x0, y0, x1, y1 = crop
    boxes = []
    # near a GT box with a high score for TB images, plus low-score clutter everywhere
    for gx0, gy0, gx1, gy1 in c.fake_gt_boxes(image_id, crop, seed):
        j = r.normal(0, 4, 4)
        bx0, by0 = max(x0, gx0 + j[0]), max(y0, gy0 + j[1])
        bx1, by1 = min(x1, gx1 + j[2]), min(y1, gy1 + j[3])
        if bx1 > bx0 and by1 > by0:
            boxes.append([bx0, by0, bx1, by1, float(r.uniform(0.45, 0.95))])
    for _ in range(int(r.integers(0, 6))):
        w, h = r.uniform(15, 70, 2)
        bx0, by0 = r.uniform(x0, x1 - w), r.uniform(y0, y1 - h)
        boxes.append([bx0, by0, bx0 + w, by0 + h, float(r.uniform(MIN_SCORE, 0.4))])
    boxes = [[round(float(v), 2) for v in b[:4]] + [round(b[4], 3)] for b in boxes]
    boxes = sorted((b for b in boxes if b[4] >= MIN_SCORE), key=lambda b: -b[4])
    return {
        "image_id": image_id,
        "crop_box": crop,
        "boxes": boxes,
        "n_lesions_above_thr": sum(b[4] >= conf_threshold for b in boxes),
        "max_score": max((b[4] for b in boxes), default=0.0),
    }


def generate(dataset: str = "tbx11k", split: str = "development", n: int = 30, seed: int = 0,
             root: Path | str | None = None) -> Path:
    doc = {
        "schema_version": c.SCHEMA_VERSION,
        "model_id": c.DET_MODEL_ID,
        "input_size": INPUT_SIZE,
        "conf_threshold": CONF_THRESHOLD,
        "coords": "full512",
        "images": [fake_image_entry(i, seed) for i in c.mock_image_ids(n, seed)],
    }
    path = c.out_path(root, "artifacts", "p3", "det", f"detections_{dataset}_{split}_v1.json")
    path.write_text(json.dumps(doc, indent=1))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="tbx11k")
    ap.add_argument("--split", default="development")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None, help="output root (default: mock/)")
    a = ap.parse_args(argv)
    print(generate(a.dataset, a.split, a.n, a.seed, a.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
