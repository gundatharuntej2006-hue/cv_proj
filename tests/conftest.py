# -*- coding: utf-8 -*-
"""Shared test setup: src/ and scripts/ on the path, the 'slow' marker, and the common fixtures."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
from tbcore.schemas import InputCXR
from mock.mock_data_generator import generate_synthetic_cxr


@pytest.fixture
def mock_cxr_image_path(tmp_path):
    out = tmp_path / "test_cxr.png"
    return generate_synthetic_cxr(str(out))


@pytest.fixture
def sample_numpy_image():
    return np.random.uniform(0.1, 0.9, (512, 512)).astype(np.float32)


@pytest.fixture
def sample_input_cxr(mock_cxr_image_path):
    return InputCXR(
        image_id="TEST_001",
        file_path=str(mock_cxr_image_path),
        width=512,
        height=512,
        channels=1,
        format="PNG"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the real RAD-DINO weights in the HF cache")
