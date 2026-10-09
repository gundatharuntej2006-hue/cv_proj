"""FIX 4: every tbcore name lives in src/p3_ops/_tbcore_adapter.py; the adapter fails loudly once a real tbcore
exists but does not provide the assumed names."""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_ops import _tbcore_adapter as ad  # noqa: E402


def test_real_tbcore_provides_every_assumed_name():
    """Skipped until tbcore exists; then it FAILS if any name in ASSUMED_API is missing (update the adapter)."""
    if not (REPO / "src" / "tbcore").exists() and not ad.tbcore_available():
        pytest.skip("tbcore (P4) not installed yet: names in ASSUMED_API are still assumptions")
    assert ad.check_api() == [], f"tbcore does not provide {ad.check_api()}: update ASSUMED_API in _tbcore_adapter.py"


def test_no_other_p3_module_imports_tbcore():
    offenders = []
    for f in (REPO / "src").glob("p3_*/**/*.py"):
        if f.name != "_tbcore_adapter.py" and re.search(r"^\s*(from tbcore|import tbcore)", f.read_text(encoding="utf-8"), re.M):
            offenders.append(f.relative_to(REPO).as_posix())
    assert offenders == []


def _fake(monkeypatch, **modules):
    pkg = types.ModuleType("tbcore")
    monkeypatch.setitem(sys.modules, "tbcore", pkg)
    for name, attrs in modules.items():
        m = types.ModuleType(f"tbcore.{name}")
        for k, v in attrs.items():
            setattr(m, k, v)
        setattr(pkg, name, m)
        monkeypatch.setitem(sys.modules, f"tbcore.{name}", m)


def test_check_api_reports_missing_names(monkeypatch):
    _fake(monkeypatch, seed={"set_all": lambda s: None}, guard={}, validators={"validate": lambda c, p: None},
          disclaimer={"DISCLAIMER": "d"})
    assert ad.check_api() == ["tbcore.guard.require_unlocked", "tbcore.disclaimer.SCOPE_HONESTY"]


def test_fail_closed_behaviour_without_tbcore(monkeypatch):
    monkeypatch.setitem(sys.modules, "tbcore", None)
    assert ad.seed_all(0) is False                         # caller seeds locally and records the fallback
    assert ad.validate("C10", "x") is False                # caller applies its own contract checks
    with pytest.raises(ad.TbcoreUnavailable):
        ad.require_unlocked("internal_test", "tbx11k", "p3_det")
    with pytest.raises(ad.TbcoreUnavailable):
        ad.disclaimer_texts()


def test_wrong_name_in_installed_tbcore_is_never_silently_skipped(monkeypatch):
    _fake(monkeypatch, validators={"check": lambda c, p: None})          # P4 used another name
    with pytest.raises(ad.TbcoreUnavailable, match="ASSUMED_API"):
        ad.validate("C10", "x")
