"""R3 `explain` (P3-09, real Grad-CAM per D19) and `render_overlay` (P3-11, p3_xai.overlay).

Frozen signatures (README 4.3):
    explain(img_for_analysis, crop_box) -> heatmap float16 [512, 512], range [0, 1], zero outside the crop
    render_overlay(img512, heatmap, boxes) -> png_path

explain() runs Grad-CAM (p3_xai.gradcam, configuration D19) on P2's classifier from
p2_cls.api.classifier_module() (R2), loaded once per process. TB_MOCK=1 serves the P3-03 stub
(deterministic Gaussian blobs) so consumers can test without RAD-DINO.

Every overlay carries the caption "attention evidence, not a finding" (drawn on the image and
stored in the PNG text metadata).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from p3_det.api import _check_inputs, _stub_rng

from .overlay import CAPTION  # noqa: F401  ("Attention evidence, not a finding"; re-exported)

IS_STUB = False            # explain (P3-09) and render_overlay (P3-11) are real
RENDER_IS_STUB = False


def _explain_stub(img: np.ndarray, crop_box: list[int]) -> np.ndarray:
    """MOCK (P3-03): deterministic Gaussian blob(s) inside the crop, normalised to [0, 1]."""
    x0, y0, x1, y1 = crop_box
    r = _stub_rng(img, [x0, y0, x1, y1], b"explain")
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    h = np.zeros((512, 512), np.float32)
    for weight in (1.0, 0.6):
        cx, cy = r.uniform(x0, x1), r.uniform(y0, y1)
        s = r.uniform(15, 50)
        h += weight * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    out = np.zeros_like(h)
    out[y0:y1, x0:x1] = h[y0:y1, x0:x1]
    out /= max(float(out.max()), 1e-12)
    return out.astype(np.float16)


def _p2_classifier():
    try:
        from p2_cls.api import classifier_module   # R2 (P2)
    except ImportError as e:
        raise RuntimeError("p2_cls.api.classifier_module (R2, P2) is not available; set TB_MOCK=1 for the "
                           "explain() mock") from e
    return classifier_module()


_provider: Callable = _p2_classifier
_explainer = None
_lock = threading.Lock()


def set_classifier_provider(provider: Callable | None) -> None:
    """Tests / sanity check: use another classifier module source (None -> P2's R2). Clears the cache."""
    global _provider, _explainer
    _provider = provider or _p2_classifier
    _explainer = None


def explainer():
    """The cached GradCamExplainer (model loaded once per process)."""
    global _explainer
    if _explainer is None:
        with _lock:
            if _explainer is None:
                import torch

                from .gradcam import GradCamExplainer
                module = _provider()
                dev = os.environ.get("TB_XAI_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
                _explainer = GradCamExplainer(module.to(dev))
    return _explainer


def explain(img_for_analysis, crop_box) -> np.ndarray:
    """Grad-CAM heatmap for the TB output: float16 [512, 512], [0, 1], zero outside crop_box."""
    img, box = _check_inputs(img_for_analysis, crop_box)
    if os.environ.get("TB_MOCK") == "1":
        return _explain_stub(img, box)
    e = explainer()
    with _lock:
        return e(img, box)


def runs_root() -> Path:
    """Base directory for generated overlay paths: TB_OVERLAY_DIR (legacy), else TB_RUNS_DIR, else <repo>/runs."""
    from p3_det.api import REPO
    return Path(os.environ.get("TB_OVERLAY_DIR") or os.environ.get("TB_RUNS_DIR") or REPO / "runs")


def render_overlay(img512, heatmap, boxes) -> str:
    """R3, frozen signature: render X-ray + Grad-CAM + detector boxes + burned-in caption; return the PNG path.

    The file goes to runs/<TB_RUN_ID or "adhoc">/overlay_<content hash>.png: named from the rendered
    content, never from a user file name. Use render_overlay_to() to choose the path or label the case.
    """
    return render_overlay_to(img512, heatmap, boxes, out_path=None, case_id=None)


def render_overlay_to(img512, heatmap, boxes, *, out_path: str | os.PathLike | None = None,
                      case_id: str | None = None) -> str:
    """render_overlay with a caller-chosen ``out_path`` and/or the salted ``case_id`` (shown in the header and
    used for the generated name runs/<run_id>/<case_id>_overlay.png). Not part of the frozen R3 interface."""
    from . import overlay
    if case_id is not None and not overlay.CASE_ID_RE.match(case_id):
        raise ValueError("case_id must be a short salted hash ([A-Za-z0-9_-]{1,64}), never a patient identifier")
    img = overlay.render(img512, heatmap, boxes, case_id=case_id)
    if out_path is None:
        run_id = os.environ.get("TB_RUN_ID", "adhoc")
        if not overlay.CASE_ID_RE.match(run_id):
            raise ValueError("TB_RUN_ID must match [A-Za-z0-9_-]{1,64}")
        out_path = overlay.default_path(runs_root(), run_id, case_id, overlay.digest_of(img))
    return str(overlay.save(img, Path(out_path)))
