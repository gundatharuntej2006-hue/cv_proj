"""MOCK detector (the P3-03 stub), used by p3_det.api.detect only when TB_MOCK=1.

Deterministic fake boxes (same input -> same output) inside the crop, so P1/P4 can build and
test without the trained checkpoint.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .result import DetResult, check_inputs

STUB_MODEL_ID = "dfine_512_stub"
STUB_CONF_THRESHOLD = 0.42   # placeholder for D18
MIN_SCORE = 0.01


def stub_rng(img: np.ndarray, box: list[int], salt: bytes) -> np.random.Generator:
    digest = hashlib.sha256(salt + np.ascontiguousarray(img).tobytes() + bytes(str(box), "ascii")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def detect_stub(img_for_analysis, crop_box) -> DetResult:
    img, (x0, y0, x1, y1) = check_inputs(img_for_analysis, crop_box)
    r = stub_rng(img, [x0, y0, x1, y1], b"detect")
    k = int(r.integers(0, 5))
    w = r.uniform(5, max(6.0, (x1 - x0) / 4), k)
    h = r.uniform(5, max(6.0, (y1 - y0) / 4), k)
    bx0 = x0 + r.uniform(0, 1, k) * np.maximum(x1 - x0 - w, 0)
    by0 = y0 + r.uniform(0, 1, k) * np.maximum(y1 - y0 - h, 0)
    boxes = np.stack([bx0, by0, np.minimum(bx0 + w, x1), np.minimum(by0 + h, y1)], axis=1) if k else np.zeros((0, 4))
    scores = r.uniform(MIN_SCORE, 0.95, k)
    order = np.argsort(-scores)
    return DetResult(
        boxes_full512=boxes[order].astype(np.float32).reshape(-1, 4),
        scores=scores[order].astype(np.float32),
        n_lesions_above_thr=int((scores >= STUB_CONF_THRESHOLD).sum()),
    )
