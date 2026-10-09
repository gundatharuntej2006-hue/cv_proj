"""P3-04: box-safe augmentation in detector input space (square input_size x input_size).

Geometry: mild isotropic scale and translation about the image centre. If any box would end up
clipped below ``min_visible`` of its area or thinner than ``min_size`` pixels, the parameters are
redrawn (up to ``max_tries``) and finally the geometric step is skipped, so augmentation never
drops, truncates heavily or degenerates a lesion box. Boxes are always clipped to the canvas.

Flips: horizontal flip is OFF by default. A mirrored chest X-ray shows the heart on the wrong
side and swaps left/right lung findings, which no real frontal CXR shows; enable it only with an
explicit decision. Vertical flips and rotations are not offered.

Photometric: mild brightness and contrast only (the C04 store is already normalised).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .geometry import clip_boxes


@dataclass(frozen=True)
class AugmentConfig:
    scale: tuple[float, float] = (0.9, 1.1)
    translate: float = 0.05          # fraction of the input size, each axis
    brightness: float = 0.10         # additive, fraction of 255
    contrast: float = 0.10           # multiplicative about the image mean
    hflip_p: float = 0.0             # off: not anatomically plausible for frontal CXR
    min_size: float = 2.0            # px in detector space
    min_visible: float = 0.7         # share of a box's area that must stay on the canvas
    max_tries: int = 10


def _affine(img: np.ndarray, s: float, tx: float, ty: float) -> np.ndarray:
    h, w = img.shape
    cx, cy = w / 2, h / 2
    # PIL wants the inverse map: input = A^-1 (output)
    coeffs = (1 / s, 0, cx - (cx + tx) / s, 0, 1 / s, cy - (cy + ty) / s)
    return np.asarray(Image.fromarray(img).transform((w, h), Image.Transform.AFFINE, coeffs,
                                                     resample=Image.Resampling.BILINEAR, fillcolor=0))


def _affine_boxes(boxes: np.ndarray, size: int, s: float, tx: float, ty: float) -> np.ndarray:
    c = size / 2
    return (boxes - c) * s + c + np.array([tx, ty, tx, ty])


def augment(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator,
            cfg: AugmentConfig = AugmentConfig()) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (image uint8 [S,S], boxes float [k,4], params). Never loses a box."""
    size = img.shape[0]
    assert img.shape == (size, size), img.shape
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    params = {"hflip": False, "scale": 1.0, "tx": 0.0, "ty": 0.0}

    if cfg.hflip_p > 0 and rng.random() < cfg.hflip_p:
        img = img[:, ::-1].copy()
        boxes = np.stack([size - boxes[:, 2], boxes[:, 1], size - boxes[:, 0], boxes[:, 3]], axis=1)
        params["hflip"] = True

    for _ in range(cfg.max_tries):
        s = float(rng.uniform(*cfg.scale))
        tx, ty = (float(v) for v in rng.uniform(-cfg.translate, cfg.translate, 2) * size)
        moved = _affine_boxes(boxes, size, s, tx, ty)
        kept, keep = clip_boxes(moved, size, size, min_size=cfg.min_size, min_visible=cfg.min_visible)
        if keep.all():
            img, boxes = _affine(img, s, tx, ty), kept
            params.update(scale=s, tx=tx, ty=ty)
            break
    # else: geometric step skipped (identity), boxes unchanged

    a = 1.0 + float(rng.uniform(-cfg.contrast, cfg.contrast))
    b = 255.0 * float(rng.uniform(-cfg.brightness, cfg.brightness))
    f = img.astype(np.float32)
    img = ((f - f.mean()) * a + f.mean() + b).clip(0, 255).astype(np.uint8)
    params.update(contrast=a, brightness=b)
    return img, boxes, params
