# -*- coding: utf-8 -*-
"""Path resolution with the TB_MOCK switch.

TEMPORARY. `tbcore.paths` (Person 4, P4-01) is the real home for this. P2 may not edit
`tbcore`, and P2-02 is due in Week 1 before `tbcore` exists, so this shim stands in and
delegates to `tbcore.paths` the moment that module is importable. Delete this file in
the PR that first imports `tbcore` successfully.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def mock_mode() -> bool:
    return os.environ.get("TB_MOCK", "") == "1"


def resolve(rel: str) -> Path:
    """Repo-relative contract path -> real path, redirected under mock/ when TB_MOCK=1."""
    try:
        from tbcore import paths as _tbcore_paths      # noqa: F401  (P4's real one)
        return Path(_tbcore_paths.resolve(rel))
    except Exception:
        pass
    rel = rel.replace("\\", "/").lstrip("/")
    return (REPO_ROOT / "mock" / rel) if mock_mode() else (REPO_ROOT / rel)
