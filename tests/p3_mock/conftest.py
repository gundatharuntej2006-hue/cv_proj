"""P3-03 test setup: import paths and fixtures; CaseResult examples live in p3_case_results.py."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "mock", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

from p3_case_results import ANALYSED, CASES, DISCLAIMER, LINK_FIRED, REJECTED, UNREADABLE  # noqa: E402,F401


@pytest.fixture(params=list(CASES))
def case_result(request):
    return copy.deepcopy(CASES[request.param])


@pytest.fixture
def img512():
    import numpy as np
    return (np.arange(512 * 512, dtype=np.uint32).reshape(512, 512) % 251).astype(np.uint8)
