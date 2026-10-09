"""Pure scanning logic behind audit checks D2, D3 and G4.

Kept separate so the audit (test_p3_audit.py) and the mutation tests (tests/p3_audit_selftest/) run the SAME
code: a mutation test plants a real violation in a temporary copy and must see the check turn FAIL.
"""
from __future__ import annotations

import re
from pathlib import Path

MARKER_RE = re.compile(r"STUB|TODO|placeholder|PLACEHOLDER|FIXME")
NOT_A_STUB_FLAG_RE = re.compile(r'\b\w*IS_STUB\s*=\s*False|"IS_STUB"')       # flags that DECLARE "not a stub"
MOCK_MODULES = {"_mock.py"}                       # TB_MOCK-only code (reported separately, see D2)
ADAPTER_MODULES = {"_tbcore_adapter.py"}          # assumed tbcore API names, one place (D2c)
HARDCODED_RE = re.compile(r"\s*[A-Za-z_]\w*\s*=\s*(0\.\d+|512|640)\b")        # assignments only


def code_markers(root: Path) -> list[str]:
    """D2a: STUB/TODO/placeholder/FIXME lines in REAL modules under root/src/p3_*/ (*.py)."""
    hits = []
    for f in sorted(Path(root).glob("src/p3_*/**/*.py")):
        if f.name in MOCK_MODULES | ADAPTER_MODULES:
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if MARKER_RE.search(line) and not NOT_A_STUB_FLAG_RE.search(line):
                hits.append(f"{f.relative_to(root).as_posix()}:{n}: {line.strip()[:110]}")
    return hits


def config_pending(root: Path) -> list[str]:
    """D2b: training-config values still awaiting the P3-01 feasibility numbers (null) or TODO comments."""
    import yaml
    out = []
    for f in sorted(Path(root).glob("src/p3_det/configs/*.yaml")):
        text = f.read_text(encoding="utf-8")
        s = yaml.safe_load(text)["schedule"]
        out += [f"{f.name}: schedule.{k} = null" for k in ("epochs", "batch_size") if s.get(k) is None]
        out += [f"{f.name}:{n}: {l.strip()[:90]}" for n, l in enumerate(text.splitlines(), 1) if MARKER_RE.search(l)]
    return out


def hardcoded_settings(api_src: str) -> list[str]:
    """D3: numeric settings assigned in p3_det/api.py (MIN_SCORE 0.01 is the C10 rule, allowed)."""
    return [l.strip() for l in api_src.splitlines() if HARDCODED_RE.match(l) and "MIN_SCORE" not in l]


def identifier_hits(texts: dict[str, str], patterns: dict[str, str], names_only: set[str]) -> dict[str, str]:
    """G4: identifier-like matches. For sources that are our own artifact NAMES/PATHS (names_only), the
    'image file name' pattern is not applied (they are PNGs by design); every other pattern is."""
    return {f"{src} / {name}": m for src, t in texts.items() for name, pat in patterns.items()
            if not (src in names_only and name == "image file name") for m in re.findall(pat, t)[:2]}
