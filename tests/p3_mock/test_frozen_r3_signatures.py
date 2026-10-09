"""FIX 3: the five R3 entry points have EXACTLY the frozen signatures of README 4.3 (no extra parameters).

Extra abilities live in separately named functions (render_overlay_to, make_pdf_with_original).
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

FROZEN_R3 = {
    "p3_det.api.detect": "(img_for_analysis, crop_box)",
    "p3_xai.api.explain": "(img_for_analysis, crop_box)",
    "p3_xai.api.render_overlay": "(img512, heatmap, boxes)",
    "p3_ops.api.make_pdf": "(case_result)",
    "p3_ops.api.log_case": "(case_result, operating_point)",
}
EXTENSIONS = {
    "p3_xai.api.render_overlay_to": "(img512, heatmap, boxes, *, out_path=None, case_id=None)",
    "p3_ops.api.make_pdf_with_original": "(case_result, *, image=None, out_path=None)",
}


def _sig(dotted: str) -> str:
    mod, fn = dotted.rsplit(".", 1)
    sig = inspect.signature(getattr(importlib.import_module(mod), fn))
    return str(sig.replace(return_annotation=inspect.Signature.empty,
                           parameters=[p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()]))


@pytest.mark.parametrize("dotted, expected", FROZEN_R3.items())
def test_frozen_r3_signature_exact(dotted, expected):
    pytest.importorskip("numpy")
    assert _sig(dotted) == expected


@pytest.mark.parametrize("dotted, expected", EXTENSIONS.items())
def test_extensions_are_separate_functions(dotted, expected):
    pytest.importorskip("numpy")
    assert _sig(dotted) == expected
