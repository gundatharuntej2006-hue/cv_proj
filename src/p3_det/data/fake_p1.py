"""P3-04: FAKE P1 data (C01, C04, C05, C06) for tests and the viz check, until P1's mocks exist.

Writes a tiny tree with the README 4.2 layouts under a root you choose (never under mock/, which
holds each producer's own mocks). Images are synthetic: dark lung fields on a lighter background
with bright lesion patches, so drawn boxes can be eyeballed. Ids look like TBX11K ids but are
generated; nothing here is real data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import MARGINS

SPLITS = ("train", "development", "calibration", "internal_test", "challenge_test", "excluded_duplicate")
_C01_FIELDS = ["image_id", "rel_path", "official_split", "split", "category", "tb_label", "active_tb_label",
               "has_boxes", "n_boxes", "dup_of", "split_version", "seed"]


def _lung_box(r) -> tuple[int, int, int, int]:
    x0, y0 = int(r.integers(70, 110)), int(r.integers(40, 70))
    return x0, y0, int(r.integers(400, 440)), int(r.integers(420, 460))


def _crop(lung, margin: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = lung
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (max(0, int(round(x0 - mx))), max(0, int(round(y0 - my))),
            min(512, int(round(x1 + mx))), min(512, int(round(y1 + my))))


def _image(r, lung, lesions) -> np.ndarray:
    img = Image.new("L", (512, 512), 170)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = lung
    mid = (x0 + x1) // 2
    d.ellipse([x0, y0, mid - 15, y1], fill=60)          # image-left lung
    d.ellipse([mid + 15, y0, x1, y1], fill=60)          # image-right lung
    d.text((x0 + 5, y0 + 5), "R", fill=255)             # side marker, makes flips visible
    for bx0, by0, bx1, by1 in lesions:
        d.ellipse([bx0, by0, bx1, by1], fill=int(r.integers(150, 220)))
    arr = np.asarray(img, np.float32) + r.normal(0, 8, (512, 512))
    return arr.clip(0, 255).astype(np.uint8)


def make_fake_p1_tree(root: str | Path, n_per_split: int = 12, seed: int = 0) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    r = np.random.default_rng(seed)
    c01, manifest, roi = [], [], []
    coco = {(s, m): {"images": [], "annotations": [], "categories": [{"id": 1, "name": "tb_lesion"}]}
            for s in SPLITS for m in MARGINS}
    ann_id = 1
    kinds = ["tb", "tb", "h", "s", "tb_nobox"]          # mix incl. TB without boxes
    for si, split in enumerate(SPLITS):
        for i in range(n_per_split):
            kind = kinds[i % len(kinds)]
            iid = f"{kind.split('_')[0]}{si:01d}{i:03d}"
            tb = kind.startswith("tb")
            lung = _lung_box(r)
            lesions = []
            if kind == "tb":
                for k in range(int(r.integers(1, 4))):
                    w, h = r.uniform(20, 80, 2)
                    if k == 0 and i % 4 == 0:            # straddles the margin-0.10 crop edge -> clipped
                        bx, by = _crop(lung, 0.10)[0] - w / 2, r.uniform(lung[1], lung[3] - h)
                    else:
                        bx, by = r.uniform(lung[0], lung[2] - w), r.uniform(lung[1], lung[3] - h)
                    lesions.append([float(max(0, bx)), float(by), float(bx + w), float(by + h)])
            img = _image(r, lung, lesions)
            rel = f"data/processed/v1/tbx11k/img512/{iid}.png"
            rel_c = f"data/processed/v1/tbx11k/img512_clahe/{iid}.png"
            for rp, a in ((rel, img), (rel_c, np.clip((img.astype(np.float32) - 128) * 1.3 + 128, 0, 255))):
                (root / rp).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(a.astype(np.uint8)).save(root / rp)
            category = {"tb": "active_tb", "tb_nobox": "uncertain_tb", "h": "healthy", "s": "sick_non_tb"}[kind]
            withheld = split == "challenge_test"
            c01.append({"image_id": iid, "rel_path": f"imgs/{iid}.png", "official_split": "train",
                        "split": split, "category": "unknown" if withheld else category,
                        "tb_label": -1 if withheld else int(tb), "active_tb_label": "",
                        "has_boxes": str(bool(lesions)), "n_boxes": len(lesions), "dup_of": "",
                        "split_version": "v1", "seed": 20260921})
            manifest.append({"image_id": iid, "dataset": "tbx11k", "path_img512": rel, "path_img512_clahe": rel_c,
                             "orig_h": 512, "orig_w": 512, "src_format": "png", "decode_status": "ok",
                             "error_msg": None, "phi_tags_removed": 0, "sha256_img512": "0" * 64})
            fallback = i % 11 == 10
            for m in MARGINS:
                crop = (0, 0, 512, 512) if fallback else _crop(lung, m)
                roi.append({"image_id": iid, "mask_path": f"data/processed/v1/masks/{iid}.png",
                            "lung_area_frac": 0.3, "n_components": 2,
                            "lung_x0": lung[0], "lung_y0": lung[1], "lung_x1": lung[2], "lung_y1": lung[3],
                            "sanity_pass": not fallback, "fail_reasons": "area" if fallback else "",
                            "margin": m, "crop_x0": crop[0], "crop_y0": crop[1], "crop_x1": crop[2],
                            "crop_y1": crop[3], "fallback_full": fallback})
                doc = coco[(split, m)]
                img_id = len(doc["images"]) + 1
                cx0, cy0, cx1, cy1 = crop
                doc["images"].append({"id": img_id, "file_name": iid, "width": cx1 - cx0, "height": cy1 - cy0,
                                      "crop_box": list(crop)})
                for bx0, by0, bx1, by1 in lesions:
                    c = [max(bx0, cx0) - cx0, max(by0, cy0) - cy0, min(bx1, cx1) - cx0, min(by1, cy1) - cy0]
                    if c[2] <= c[0] or c[3] <= c[1]:
                        continue                          # fully outside: C06 drops it
                    doc["annotations"].append({
                        "id": ann_id, "image_id": img_id, "bbox": [c[0], c[1], c[2] - c[0], c[3] - c[1]],
                        "area": (c[2] - c[0]) * (c[3] - c[1]), "category_id": 1, "iscrowd": 0,
                        "tb_subtype": "active", "orig_bbox_512": [bx0, by0, bx1 - bx0, by1 - by0],
                        "clipped": c != [bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0]})
                    ann_id += 1

    (root / "data/splits").mkdir(parents=True, exist_ok=True)
    with open(root / "data/splits/tbx11k_splits_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_C01_FIELDS)
        w.writeheader()
        w.writerows(c01)
    pq.write_table(pa.Table.from_pylist(manifest), root / "data/processed/v1/manifest.parquet")
    pq.write_table(pa.Table.from_pylist(roi), root / "data/processed/v1/lung_roi.parquet")
    (root / "data/boxes/v1").mkdir(parents=True, exist_ok=True)
    for (split, m), doc in coco.items():
        (root / f"data/boxes/v1/coco_{split}_m{m:.2f}.json").write_text(json.dumps(doc))
    return root
