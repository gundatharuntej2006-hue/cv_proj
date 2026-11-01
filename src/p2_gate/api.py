# -*- coding: utf-8 -*-
"""R2, gate half:  gate(img512) -> GateResult

NO LONGER A STUB. With a bundle at `artifacts/p2/bundle/` this runs the real gate: the D07
quality rules derived in P2-05 from a synthetic degradation sweep, and the D13 OOD score
from P2-11 — k=5 mean cosine distance to a bank of training embeddings, thresholded
conformally. `is_stub` is then False and P4-09's gate will accept the result.

With no bundle on disk — a fresh clone, or CI — the Week-1 stub behaviour is unchanged:
genuine quality rules with provisional thresholds, and an OOD *hash* standing in for a
distance, flagged `is_stub=True`. TB_P2_GATE_FORCE_STUB=1 pins that path.

Two properties of the real gate are deliberate and must not be "fixed":

  1. **The OOD threshold may be None, and then nothing is flagged.** `ood.fit_threshold`
     abstains when the calibration set is too small for the conformal guarantee to exist
     (fewer than 1/alpha - 1 images). A gate that cannot calibrate must not reject
     everything, so it rejects nothing and says `ood_abstained=True`. See
     `docs/p2_ood_gate_finding.md` for why that matters: the threshold does not transfer
     across sites, so at a new hospital the honest state really is "not calibrated yet".
  2. **Only the four C09 reasons are ever emitted.** Any other string is a contract
     violation downstream, so the reasons are filtered against the list, not trusted.

The embedding step makes the real gate ~100x slower than the stub (a ViT-B/14 forward), so
`gate()` takes an optional precomputed embedding. Person 4's `run_case` already embeds the
image for the classifier; pass it in rather than paying twice.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# The four reasons in the C09 contract. Nothing outside this list may ever be emitted.
QUALITY_REASONS = ["blank", "under_exposed", "over_exposed", "heavily_cropped"]

STUB_OOD_THRESHOLD = 0.30       # a placeholder; D13 sets the real one on development
BUNDLE_DIR = Path(__file__).resolve().parents[2] / "artifacts/p2/bundle"

_CACHE = {}


@dataclass
class GateResult:
    """R2 return type, matching the C09 columns one for one."""
    quality_pass: bool
    reasons: List[str]
    ood_score: float
    ood_threshold: Optional[float]
    gate_pass: bool
    is_stub: bool = field(default=True)
    ood_abstained: bool = field(default=False)


def _bundle(bundle_dir=None):
    """(OodBank, threshold) from the bundle, or None when there is no bundle on disk."""
    if os.environ.get("TB_P2_GATE_FORCE_STUB") == "1":
        return None
    d = Path(bundle_dir or BUNDLE_DIR)
    key = str(d)
    if key in _CACHE:
        return _CACHE[key]
    meta_p, bank_p = d / "bundle.json", d / "bank.npz"
    if not (meta_p.exists() and bank_p.exists()):
        return None
    import json
    from p2_gate import ood as O
    meta = json.load(open(meta_p, encoding="utf-8"))
    with np.load(bank_p) as z:
        bank = z["bank"].astype(np.float32)
    _CACHE[key] = (O.OodBank(bank, k=meta.get("ood_k", 5)), meta.get("ood_threshold"))
    return _CACHE[key]


def is_stub():
    """True when no bank is available, so a caller can check before running a batch."""
    return _bundle() is None


def _stub_quality(img):
    """Provisional rules, kept for the no-bundle path. Thresholds here are guesses; D07's
    are derived. The point is that the function reads the pixels and can actually fail."""
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


def _as_uint8_gray(img):
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.5 else a, 0, 255).astype(np.uint8)
    return a


def _embedder():
    from p2_cls.embed import Embedder
    if "embedder" not in _CACHE:
        _CACHE["embedder"] = Embedder()
    return _CACHE["embedder"]


def gate(img512, embedding=None, bundle_dir=None):
    """R2. img512 is a uint8 512x512 image (grayscale or 3-channel).

    `embedding` is an optional precomputed C07-style embedding for this image (L2-normalised,
    as `embed.Embedder.embed` returns). Pass it to avoid a second backbone forward pass;
    omit it and the gate embeds the image itself.
    """
    b = _bundle(bundle_dir)

    if b is None:
        qp, reasons = _stub_quality(img512)
        score = _stub_ood_score(img512)
        return GateResult(quality_pass=qp, reasons=_clean(reasons), ood_score=score,
                          ood_threshold=STUB_OOD_THRESHOLD,
                          gate_pass=bool(qp and score <= STUB_OOD_THRESHOLD),
                          is_stub=True, ood_abstained=False)

    from p2_gate import quality as Q
    bank, thr = b
    qr = Q.assess(_as_uint8_gray(img512))
    reasons = _clean(list(qr.reasons))
    if not qr.passed:
        # a failed image is not embedded: the score would be meaningless and it costs a
        # forward pass we do not need to spend to reject the case
        return GateResult(quality_pass=False, reasons=reasons, ood_score=float("nan"),
                          ood_threshold=thr, gate_pass=False, is_stub=False,
                          ood_abstained=thr is None)

    if embedding is None:
        emb = _embedder().embed([_as_uint8_gray(img512)]).astype(np.float32)
    else:
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    score = float(bank.score(emb)[0])
    # thr is None when the conformal guarantee does not exist on this calibration set;
    # a gate that cannot calibrate flags nothing rather than rejecting everything
    ood_fail = bool(thr is not None and score > thr)
    return GateResult(quality_pass=True, reasons=reasons, ood_score=score,
                      ood_threshold=thr, gate_pass=not ood_fail, is_stub=False,
                      ood_abstained=thr is None)


def _clean(reasons):
    """C09 allows exactly four reason strings. Filter rather than trust: a new rule added
    upstream must not leak a fifth value into Person 4's parquet."""
    out = [r for r in reasons if r in QUALITY_REASONS]
    bad = [r for r in reasons if r not in QUALITY_REASONS]
    if bad:
        raise ValueError("quality reasons outside the C09 contract: %s" % bad)
    return out
