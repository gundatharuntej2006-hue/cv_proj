"""FIX 4: the REAL (non-mock) R3 paths contain no placeholder returns.

1. Static: no function in a real module of src/p3_* has a body that is only `pass`, `...`, `return None`
   or `raise NotImplementedError` (mock module and tbcore adapter excluded).
2. Dynamic: with TB_MOCK unset and upstream missing, the real entry points RAISE instead of returning
   fake output.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
EXCLUDED = {"_mock.py", "_tbcore_adapter.py"}


def _placeholder_functions(path: Path, base: Path = REPO) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant)
                                             and isinstance(b.value.value, str))]          # drop docstring
        if len(body) != 1:
            continue
        b = body[0]
        trivial = isinstance(b, ast.Pass) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                              and b.value.value is Ellipsis) \
            or (isinstance(b, ast.Return) and (b.value is None or (isinstance(b.value, ast.Constant) and b.value.value is None))) \
            or (isinstance(b, ast.Raise) and "NotImplementedError" in ast.dump(b))
        if trivial:
            out.append(f"{path.relative_to(base).as_posix()}:{node.lineno} {node.name}()")
    return out


def test_no_placeholder_function_bodies_in_real_modules():
    found = [h for f in sorted((REPO / "src").glob("p3_*/**/*.py")) if f.name not in EXCLUDED
             for h in _placeholder_functions(f)]
    assert found == []


def test_static_check_can_fail(tmp_path):
    f = tmp_path / "m.py"
    f.write_text('def detect(img, crop):\n    """doc"""\n    return None\n\n\ndef ok():\n    return 1\n')
    assert _placeholder_functions(f, tmp_path) == ["m.py:1 detect()"]


def test_real_entry_points_raise_instead_of_faking(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    monkeypatch.delenv("TB_MOCK", raising=False)
    monkeypatch.setenv("TB_DECISIONS_DIR", str(tmp_path / "no_decisions"))
    img = np.zeros((512, 512), np.uint8)
    from p3_det import api as det_api
    det_api.reset_cache()
    with pytest.raises(RuntimeError):
        det_api.detect(img, [0, 0, 512, 512])                       # no locked decisions -> refuses
    import importlib.util
    if importlib.util.find_spec("p2_cls") is None:
        from p3_xai import api as xai_api
        xai_api.set_classifier_provider(None)
        with pytest.raises(RuntimeError):
            xai_api.explain(img, [0, 0, 512, 512])                  # no P2 classifier -> refuses
    if importlib.util.find_spec("tbcore") is None:
        from p3_ops import api as ops_api
        sys.path.insert(0, str(REPO / "tests" / "p3_mock"))
        from p3_case_results import ANALYSED
        with pytest.raises(RuntimeError, match="tbcore"):
            ops_api.make_pdf(ANALYSED)                              # no single-source disclaimer -> refuses
