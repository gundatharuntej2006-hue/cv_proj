"""P3-10: localisation agreement between a Grad-CAM heatmap and annotated lesion boxes (methodology 12.5).

Both work on a heatmap in full 512 coordinates and ALL original lesion boxes of the image ([x0, y0,
x1, y1], 512 coordinates). A pixel (x, y) belongs to a box when its centre (x + 0.5, y + 0.5) lies
inside it; overlapping boxes count once (union).

* pointing_hit  - plain language: "does the single hottest pixel of the heatmap land on a lesion?"
                  Peak = first maximum in row-major order. A heatmap with no mass has no peak:
                  pointing_hit is False.
* energy_in_box - plain language: "what share of the heatmap's total heat lies on lesions?"
                  sum(heat inside the boxes) / sum(all heat), float32 in [0, 1]; 0.0 for a
                  heatmap with no mass.

These are agreement with incomplete radiologist boxes, not proof of clinical correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUBTYPE_FROM_C01 = {"active_tb": "active", "latent_tb": "latent", "active_latent_tb": "active+latent"}


def box_mask(boxes, shape=(512, 512)) -> np.ndarray:
    """Union of boxes as a boolean pixel mask (pixel centres inside a box)."""
    h, w = shape
    mask = np.zeros(shape, bool)
    for x0, y0, x1, y1 in np.asarray(boxes, float).reshape(-1, 4):
        c0, c1 = int(np.ceil(x0 - 0.5)), int(np.floor(x1 - 0.5))
        r0, r1 = int(np.ceil(y0 - 0.5)), int(np.floor(y1 - 0.5))
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, w - 1), min(r1, h - 1)
        if c1 >= c0 and r1 >= r0:
            mask[r0:r1 + 1, c0:c1 + 1] = True
    return mask


def peak(heatmap: np.ndarray) -> tuple[int, int]:
    """(peak_x, peak_y) of the first maximum in row-major order."""
    h = np.asarray(heatmap, np.float32)
    y, x = np.unravel_index(int(np.argmax(h)), h.shape)
    return int(x), int(y)


def pointing_hit(heatmap: np.ndarray, boxes) -> bool:
    h = np.asarray(heatmap, np.float32)
    if not np.isfinite(h).all() or h.max() <= 0:
        return False
    x, y = peak(h)
    return bool(box_mask(boxes, h.shape)[y, x])


def energy_in_box(heatmap: np.ndarray, boxes) -> np.float32:
    h = np.clip(np.nan_to_num(np.asarray(heatmap, np.float64)), 0, None)
    total = h.sum()
    if total <= 0:
        return np.float32(0.0)
    return np.float32(h[box_mask(boxes, h.shape)].sum() / total)


@dataclass(frozen=True)
class LocRow:
    image_id: str
    model_id: str
    peak_x: int
    peak_y: int
    pointing_hit: bool
    energy_in_box: float
    n_gt_boxes: int
    tb_subtype: str | None


def localisation_row(image_id: str, model_id: str, heatmap: np.ndarray, boxes, tb_subtype: str | None) -> LocRow:
    b = np.asarray(boxes, float).reshape(-1, 4)
    if not len(b):
        raise ValueError(f"{image_id}: localisation is computed only on TB images with annotated boxes")
    x, y = peak(heatmap)
    return LocRow(image_id, model_id, x, y, pointing_hit(heatmap, b), float(energy_in_box(heatmap, b)), len(b),
                  tb_subtype)
