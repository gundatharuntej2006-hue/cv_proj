"""P3-01: decision D20 (decisions/p3_licences.json) against the C16 decision schema."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
D20 = REPO / "decisions" / "p3_licences.json"

from c16_standin import C16_STANDIN, CONTRACT_SCHEMA  # noqa: E402


@pytest.fixture(scope="module")
def d20():
    return json.loads(D20.read_text(encoding="utf-8"))


def test_valid_against_c16_schema(d20):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(CONTRACT_SCHEMA.read_text()) if CONTRACT_SCHEMA.exists() else C16_STANDIN
    jsonschema.validate(d20, schema, format_checker=jsonschema.FormatChecker())


def test_d20_fields(d20):
    assert d20["decision_id"] == "D20" and d20["owner"] == "p3"
    assert d20["data_used"] == ["none"]
    assert d20["reviewed_by"] == "p4"
    assert set(d20["value"]) == {"dfine", "pytorch_grad_cam", "rtdetrv2"}
    assert {k: v["licence"] for k, v in d20["value"].items()} == {
        "dfine": "Apache-2.0", "pytorch_grad_cam": "MIT", "rtdetrv2": "Apache-2.0"}


def test_lock_fields_consistent(d20):
    if d20["locked"]:
        assert d20["locked_at"] and d20["git_sha"]
        assert d20["locked_at"] >= d20["preregistered_at"]
    else:
        assert d20["locked_at"] is None and d20["git_sha"] is None


def test_dfine_pin_matches_makefile(d20):
    mk = (REPO / "mk" / "p3.mk").read_text()
    pinned = re.search(r"P3_DFINE_COMMIT := (\w+)", mk).group(1)
    assert d20["value"]["dfine"]["commit"] == pinned


def test_dfine_licence_hash_matches_checkout(d20):
    lic = REPO / "third_party" / "D-FINE" / "LICENSE"
    if not lic.exists():
        pytest.skip("D-FINE submodule not checked out")
    assert hashlib.sha256(lic.read_bytes()).hexdigest() == d20["value"]["dfine"]["licence_sha256"]
    assert "Apache License" in lic.read_text() and "Version 2.0" in lic.read_text()
