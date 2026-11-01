# -*- coding: utf-8 -*-
"""Put src/ on the path for the test run. Replaced by the package install (P1's
pyproject.toml) once the real repo skeleton exists."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the real RAD-DINO weights in the HF cache")
