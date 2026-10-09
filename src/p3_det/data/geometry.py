"""P3-04: box geometry between full-image 512 coordinates, the lung crop and detector input space.

Conventions (all boxes are float [x0, y0, x1, y1], pixel-edge coordinates):
  * full512 : the 512x512 preprocessed image (C04), as used by C05 crop boxes, C06 orig_bbox_512 and C10.
  * crop    : origin at (crop_x0, crop_y0); size (x1 - x0) x (y1 - y0); crop_box is end-exclusive.
  * det     : the crop resized (no aspect preservation, same as D-FINE's Resize op) to input_size^2.

Forward and inverse mappings are exact affine maps, so full512 -> det -> full512 round-trips to
floating-point precision; the only pixel-level error comes from image interpolation.
"""
from __future__ import annotations

import numpy as np

FULL = 512


def as_boxes(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return b


def xywh_to_xyxy(xywh) -> np.ndarray:
    b = np.asarray(xywh, dtype=np.float64).reshape(-1, 4).copy()
    b[:, 2:] += b[:, :2]
    return b


def check_crop_box(crop_box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in crop_box)
    if not (0 <= x0 < x1 <= FULL and 0 <= y0 < y1 <= FULL):
        raise ValueError(f"crop_box must satisfy 0 <= x0 < x1 <= 512, 0 <= y0 < y1 <= 512; got {crop_box}")
    return x0, y0, x1, y1


def clip_boxes(boxes, width: float, height: float, min_size: float = 1.0,
               min_visible: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Clip to [0, width] x [0, height]; return (kept boxes, keep mask over the input).

    A box is dropped when it is fully outside, thinner than ``min_size`` after clipping, or
    keeps less than ``min_visible`` of its original area.
    """
    b = as_boxes(boxes)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, width)
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, height)
    w, h = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    keep = (w >= min_size) & (h >= min_size)
    if min_visible > 0:
        keep &= (w * h) >= min_visible * np.maximum(area, 1e-12)
    return c[keep], keep


def full512_to_crop(boxes, crop_box, min_size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Shift full512 boxes into crop coordinates, clip to the crop, drop boxes fully outside."""
    x0, y0, x1, y1 = check_crop_box(crop_box)
    b = as_boxes(boxes) - np.array([x0, y0, x0, y0], dtype=np.float64)
    return clip_boxes(b, x1 - x0, y1 - y0, min_size=min_size)


def crop_scale(crop_box, input_size: int) -> tuple[float, float]:
    x0, y0, x1, y1 = check_crop_box(crop_box)
    return input_size / (x1 - x0), input_size / (y1 - y0)


def crop_to_det(boxes, crop_box, input_size: int) -> np.ndarray:
    sx, sy = crop_scale(crop_box, input_size)
    return as_boxes(boxes) * np.array([sx, sy, sx, sy])


def full512_to_det(boxes, crop_box, input_size: int, min_size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """full512 -> crop (shift, clip, drop outside) -> detector input space."""
    c, keep = full512_to_crop(boxes, crop_box, min_size=min_size)
    return crop_to_det(c, crop_box, input_size), keep


def det_to_full512(boxes_det, crop_box, input_size: int, clip: bool = True) -> np.ndarray:
    """Inverse mapping for C10 / R3: detector-space boxes -> full 512 image coordinates.

    Boxes are clipped to the crop (the detector never saw anything outside it).
    """
    x0, y0, x1, y1 = check_crop_box(crop_box)
    sx, sy = crop_scale(crop_box, input_size)
    b = as_boxes(boxes_det) / np.array([sx, sy, sx, sy]) + np.array([x0, y0, x0, y0])
    if clip:
        b[:, [0, 2]] = b[:, [0, 2]].clip(x0, x1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(y0, y1)
    return b
