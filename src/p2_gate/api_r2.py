# -*- coding: utf-8 -*-
"""R2, gate half:  gate(img512) -> GateResult

STUB UNTIL W6. The signature is frozen at `contracts-v1`; the rules are not.

The real quality rules come from P2-05 (D07, synthetic degradations confirmed on the
development split) and the real OOD score from P2-11 (D13, k=5 mean cosine distance to a
bank of one unaugmented embedding per training image). Neither exists in Week 1, so this
stub gives Person 4's `run_case` something with the right shape and deterministic
behaviour to integrate against.

The quality part is a crude but genuine measurement of the image, not a random draw, so a
blank or blown-out test image really is rejected and P4's "gate rejected" fixture behaves
like the real thing. The OOD part is a deterministic function of image content and is
**not** a distance to anything; it exists only so the field is populated. `is_stub=True`
says so, and the P4-09 gate must refuse any reported number carrying that flag.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

# The four reasons in the C09 contract. Nothing outside this list may ever be emitted.
QUALITY_REASONS = ["blank", "under_exposed", "over_exposed", "heavily_cropped"]

STUB_OOD_THRESHOLD = 0.30       # a placeholder; D13 sets the real one on development


@dataclass
class GateResult:
    """R2 return type, matching the C09 columns one for one."""
    quality_pass: bool
    reasons: List[str]
    ood_score: float
    ood_threshold: float
    gate_pass: bool
    is_stub: bool = field(default=True)


def _quality(img):
    """Provisional rules. Thresholds here are guesses and are replaced wholesale by D07;
    the point today is that the function reads the pixels and can actually fail."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32)
    if a.max() > 1.5:                       # accept both uint8 and already-scaled input
        a = a / 255.0
    reasons = []
    if float(a.std()) < 0.02:
        reasons.append("blank")
    else:
        if float(a.mean()) < 0.12:
            reasons.append("under_exposed")
        if float(a.mean()) > 0.88 or float((a > 0.98).mean()) > 0.5:
            reasons.append("over_exposed")
    h, w = a.shape[:2]
    if min(h, w) < 256 or (max(h, w) / max(min(h, w), 1)) > 1.6:
        reasons.append("heavily_cropped")
    return (len(reasons) == 0), reasons


def _stub_ood_score(img):
    """Deterministic in the image content, so the same image always scores the same and
    P4's fixtures are reproducible. It is a hash, not a distance: see the module note."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    small = a[::max(a.shape[0] // 16, 1), ::max(a.shape[1] // 16, 1)][:16, :16]
    digest = int.from_bytes(np.ascontiguousarray(
        np.round(small * 255).astype(np.uint8)).tobytes()[:8], "little", signed=False)
    return float((digest % 10_000) / 10_000.0 * 0.6)      # in [0, 0.6)


def gate(img512):
    """R2. img512 is a uint8 512x512 image (grayscale or 3-channel)."""
    qp, reasons = _quality(img512)
    score = _stub_ood_score(img512)
    ood_pass = score <= STUB_OOD_THRESHOLD
    return GateResult(quality_pass=qp, reasons=reasons, ood_score=score,
                      ood_threshold=STUB_OOD_THRESHOLD, gate_pass=bool(qp and ood_pass),
                      is_stub=True)
