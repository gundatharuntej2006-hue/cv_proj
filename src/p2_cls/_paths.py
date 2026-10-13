# -*- coding: utf-8 -*-
"""Path resolution for P2 modules. Delegates to tbcore.paths (P4's shared module), which
exists in this repo; the TB_MOCK redirection lives there."""
from pathlib import Path
from tbcore.paths import get_repo_root, resolve_path, is_mock_mode

REPO_ROOT = get_repo_root()


def mock_mode() -> bool:
    return is_mock_mode()


def resolve(rel: str) -> Path:
    return resolve_path(rel)
