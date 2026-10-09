"""The P3 audit runs only on request (P3_VERIFY=1, set by `make p3-verify` / p3_ops.verify_all), never as part
of the normal test suite or CI (its A3 check itself runs the whole suite)."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("P3_VERIFY") == "1":
        return
    skip = pytest.mark.skip(reason="P3 audit: run with make p3-verify")
    for item in items:
        if "p3_verify" in str(item.fspath):
            item.add_marker(skip)
