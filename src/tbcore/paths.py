"""Path resolution utilities supporting TB_MOCK=1 redirection."""

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Returns absolute path to the repository root."""
    # This file is located at <root>/src/tbcore/paths.py
    return Path(__file__).resolve().parent.parent.parent


def is_mock_mode() -> bool:
    """Returns True if TB_MOCK environment variable is set to '1' or 'true'."""
    val = os.environ.get("TB_MOCK", "").strip().lower()
    return val in ("1", "true", "yes")


def resolve_path(rel_path: str | Path, allow_mock: bool = True) -> Path:
    """Resolves a relative path within the repo root, redirecting to mock/ if TB_MOCK=1."""
    root = get_repo_root()
    p = Path(rel_path)
    if p.is_absolute():
        return p

    if allow_mock and is_mock_mode():
        mock_p = root / "mock" / p
        if mock_p.exists():
            return mock_p

    return root / p


def get_data_dir() -> Path:
    return resolve_path("data")


def get_artifacts_dir() -> Path:
    return resolve_path("artifacts")


def get_runs_dir() -> Path:
    return resolve_path("runs")


def get_decisions_dir() -> Path:
    return resolve_path("decisions", allow_mock=False)


def get_contracts_dir() -> Path:
    return resolve_path("contracts", allow_mock=False)
