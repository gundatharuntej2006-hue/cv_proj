"""P3-03: shared helpers for the P3 mock generators (C10, C11, C15). MOCK DATA ONLY.

Mocks sit at the contract's relative path under ``mock/`` (README 4.1), e.g.
``mock/artifacts/p3/det/detections_tbx11k_development_v1.json``. Image ids look like TBX11K ids
(``tb0001``, ``s0002``, ``h0003``) but are generated and refer to no real image or patient.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MOCK_ROOT = REPO / "mock"
SCHEMA_VERSION = "1.0"
DET_MODEL_ID = "dfine_512"                        # C14 example
CAM_MODEL_ID = "radino_crop_m0.10_clahe0_bin"     # C11 / C12 example
_PREFIXES = ("tb", "s", "h")                      # TB, sick-non-TB, healthy


def mock_image_ids(n: int, seed: int = 0) -> list[str]:
    """Deterministic fake image ids; every third is TB so positives and negatives both appear."""
    return [f"{_PREFIXES[i % 3]}{seed * 10000 + i + 1:04d}" for i in range(n)]


def is_tb(image_id: str) -> bool:
    return image_id.startswith("tb")


def rng_for(image_id: str, seed: int, salt: int) -> np.random.Generator:
    """Per-image generator so C10 and C11 mocks agree on crop boxes for the same image."""
    return np.random.default_rng([seed, salt, *image_id.encode()])


def fake_crop_box(image_id: str, seed: int = 0) -> list[int]:
    """Lung crop [x0, y0, x1, y1] in 512 coordinates, like C05/C14 (e.g. [10, 0, 507, 493])."""
    r = rng_for(image_id, seed, salt=1)
    x0, y0 = (int(v) for v in r.integers(0, 40, 2))
    x1, y1 = (int(v) for v in r.integers(470, 513, 2))
    return [x0, y0, x1, y1]


def fake_gt_boxes(image_id: str, crop_box: list[int], seed: int = 0) -> list[list[float]]:
    """Fake annotated lesion boxes [x0, y0, x1, y1] inside the crop (TB images only)."""
    if not is_tb(image_id):
        return []
    r = rng_for(image_id, seed, salt=2)
    x0, y0, x1, y1 = crop_box
    boxes = []
    for _ in range(int(r.integers(1, 4))):
        w, h = r.uniform(25, 90, 2)
        bx, by = r.uniform(x0, x1 - w), r.uniform(y0, y1 - h)
        boxes.append([round(float(bx), 1), round(float(by), 1), round(float(bx + w), 1), round(float(by + h), 1)])
    return boxes


def out_path(root: Path | str | None, *parts: str) -> Path:
    p = Path(root or MOCK_ROOT).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
