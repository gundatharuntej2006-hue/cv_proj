"""R3 detect() result type and input checks (shared by the real detector and the mock)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetResult:
    """Detector output in full-image 512 coordinates.

    boxes_full512: float32 [k, 4] as [x0, y0, x1, y1], sorted by descending score
    scores:        float32 [k], every score >= 0.01
    n_lesions_above_thr: number of boxes with score >= the locked threshold (D18)
    """
    boxes_full512: np.ndarray
    scores: np.ndarray
    n_lesions_above_thr: int


def check_inputs(img_for_analysis, crop_box) -> tuple[np.ndarray, list[int]]:
    img = np.asarray(img_for_analysis)
    if img.ndim not in (2, 3) or img.shape[:2] != (512, 512):
        raise ValueError(f"img_for_analysis must be 512x512 (HxW or HxWxC), got {img.shape}")
    box = [int(v) for v in crop_box]
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512):
        raise ValueError(f"crop_box must satisfy 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512, got {box}")
    return img, box
