"""P3-10: all original lesion boxes of a split in full 512 coordinates (D21: evaluate against ALL boxes).

Preferred source: an uncropped box file ``data/boxes/v1/coco_{split}_full.json`` (requested from P1,
docs/p3_contract_notes.md item 11). Until it exists, the widest-margin C06 file is used: it drops
only boxes lying fully outside the widest lung crop, and a warning says so.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .config import MARGINS, DataPaths


def original_boxes(paths: DataPaths, split: str) -> dict[str, dict]:
    """{image_id: {"boxes": float [k, 4] xyxy (512 coords), "subtypes": [str | None, ...]}}"""
    full = paths.root / f"data/boxes/v1/coco_{split}_full.json"
    if full.exists():
        src = full
    else:
        src = paths.coco(split, max(MARGINS))
        warnings.warn(f"no {full.name}; using {src.name}: boxes fully outside the widest crop are missing",
                      stacklevel=2)
    doc = json.loads(Path(src).read_text())
    name = {im["id"]: im["file_name"] for im in doc["images"]}
    out = {n: {"boxes": [], "subtypes": []} for n in name.values()}
    for a in doc["annotations"]:
        x, y, w, h = a.get("orig_bbox_512", a["bbox"])
        e = out[name[a["image_id"]]]
        e["boxes"].append([x, y, x + w, y + h])
        e["subtypes"].append(a.get("tb_subtype"))
    for e in out.values():
        e["boxes"] = np.asarray(e["boxes"], float).reshape(-1, 4)
    return out
