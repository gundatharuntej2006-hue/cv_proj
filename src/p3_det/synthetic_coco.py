"""P3-01: tiny synthetic COCO dataset for detector feasibility runs.

Mirrors the shape of contract C06 (single category {id: 1, name: "tb_lesion"},
non-TB images kept as negatives with no annotations) but is pure noise + blobs:
no real images, no patient identifiers. Image ids are ``syn_NNNNN``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CATEGORY = {"id": 1, "name": "tb_lesion"}


def make_synthetic_coco(
    out_dir: str | Path,
    n_images: int,
    size: int = 512,
    seed: int = 0,
    frac_positive: float = 0.6,
    max_boxes: int = 4,
    split: str = "train",
) -> Path:
    """Write ``out_dir/{split}/*.png`` and ``out_dir/{split}.json``; return the json path."""
    rng = np.random.default_rng(seed)
    img_dir = Path(out_dir) / split
    img_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    for i in range(1, n_images + 1):
        file_name = f"syn_{i:05d}.png"
        arr = rng.normal(90, 20, (size, size)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        draw = ImageDraw.Draw(img)
        n_boxes = int(rng.integers(1, max_boxes + 1)) if rng.random() < frac_positive else 0
        for _ in range(n_boxes):
            w, h = (float(v) for v in rng.uniform(0.04, 0.2, 2) * size)
            x, y = float(rng.uniform(0, size - w)), float(rng.uniform(0, size - h))
            draw.ellipse([x, y, x + w, y + h], fill=int(rng.integers(160, 230)))
            annotations.append({
                "id": ann_id, "image_id": i, "bbox": [x, y, w, h], "area": w * h, "category_id": CATEGORY["id"], "iscrowd": 0,
            })
            ann_id += 1
        img.convert("RGB").save(img_dir / file_name)
        images.append({"id": i, "file_name": file_name, "width": size, "height": size})
    ann_path = Path(out_dir) / f"{split}.json"
    ann_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": [CATEGORY]}))
    return ann_path
