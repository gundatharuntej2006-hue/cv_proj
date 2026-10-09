"""P3-08: C10 detections file: build, validate, write (README 4.2, contract C10).

``validate_c10`` uses tbcore's validator when P4 has published it; until then it applies the C10
table directly (types, coords == "full512", boxes inside [0, 512], score >= 0.01, counts consistent).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
MIN_SCORE = 0.01
NAME_RE = re.compile(r"^detections_[a-z0-9]+_[a-z_]+_v\d+\.json$")


def entry(image_id: str, crop_box, boxes_full512: np.ndarray, scores: np.ndarray, conf_threshold: float) -> dict:
    """One images[] element. Coordinates rounded to 0.001 px; scores written unrounded so
    n_lesions_above_thr and max_score are exactly reproducible from the file."""
    boxes = [[*(round(float(v), 3) for v in b), float(s)] for b, s in zip(boxes_full512, scores)]
    return {
        "image_id": image_id,
        "crop_box": [int(v) for v in crop_box],
        "boxes": boxes,
        "n_lesions_above_thr": int(sum(b[4] >= conf_threshold for b in boxes)),
        "max_score": max((b[4] for b in boxes), default=0.0),
    }


def document(model_id: str, input_size: int, conf_threshold: float, images: list[dict]) -> dict:
    return {"schema_version": SCHEMA_VERSION, "model_id": model_id, "input_size": int(input_size),
            "conf_threshold": float(conf_threshold), "coords": "full512", "images": images}


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"C10 invalid: {msg}")


def validate_doc(doc: dict) -> None:
    """Stand-in for tbcore's C10 validator (README 4.2 C10 table)."""
    for k, t in (("schema_version", str), ("model_id", str), ("input_size", int), ("conf_threshold", float),
                 ("coords", str), ("images", list)):
        _check(isinstance(doc.get(k), t) and not isinstance(doc.get(k), bool), f"{k} must be {t.__name__}")
    _check(doc["input_size"] in (512, 640), "input_size must be 512 or 640")
    _check(doc["coords"] == "full512", "coords must be 'full512'")
    _check(0.0 <= doc["conf_threshold"] <= 1.0, "conf_threshold out of [0, 1]")
    seen = set()
    for im in doc["images"]:
        iid = im.get("image_id")
        _check(isinstance(iid, str) and iid and iid not in seen, f"bad or duplicate image_id {iid!r}")
        seen.add(iid)
        cb = im.get("crop_box")
        _check(isinstance(cb, list) and len(cb) == 4 and all(isinstance(v, int) for v in cb), f"{iid}: crop_box")
        _check(0 <= cb[0] < cb[2] <= 512 and 0 <= cb[1] < cb[3] <= 512, f"{iid}: crop_box out of range")
        scores = []
        for b in im.get("boxes", []):
            _check(isinstance(b, list) and len(b) == 5, f"{iid}: box must be [x0,y0,x1,y1,score]")
            x0, y0, x1, y1, s = b
            _check(0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512, f"{iid}: box {b[:4]} outside [0,512] or empty")
            _check(MIN_SCORE <= s <= 1.0, f"{iid}: score {s} < 0.01 or > 1")
            scores.append(s)
        _check(im.get("n_lesions_above_thr") == sum(s >= doc["conf_threshold"] for s in scores),
               f"{iid}: n_lesions_above_thr inconsistent")
        _check(im.get("max_score") == (max(scores) if scores else 0.0), f"{iid}: max_score inconsistent")


def validate_c10(path: Path) -> None:
    path = Path(path)
    _check(bool(NAME_RE.match(path.name)), f"file name {path.name} does not follow detections_{{dataset}}_{{split}}_v1.json")
    try:
        from tbcore import validators  # P4 (P4-01)
    except ImportError:
        validate_doc(json.loads(path.read_text()))
        return
    validators.validate("C10", path)   # API name to confirm with P4; a mismatch raises, never skips


def write(path: Path, doc: dict) -> Path:
    validate_doc(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    os.replace(tmp, path)
    validate_c10(path)
    return path
