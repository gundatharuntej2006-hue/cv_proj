"""FIX 1: the C11 mock can never drift from the real C11 writer (columns, dtypes, TB-with-box rows only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "mock"):
    sys.path.insert(0, str(p))

import make_c11  # noqa: E402

from p3_xai import c11  # noqa: E402


def test_mock_uses_the_real_writer_and_defines_no_schema_of_its_own():
    src = (REPO / "mock" / "make_c11.py").read_text(encoding="utf-8")
    assert "c11.write_table(" in src and "pa.schema(" not in src


def test_generated_mock_matches_real_c11_columns_dtypes_and_row_rule(tmp_path):
    import pyarrow.parquet as pq
    path = make_c11.generate(n=24, seed=1, root=tmp_path)
    t = pq.read_table(path)
    assert t.schema.names == list(c11.COLUMNS)
    assert [f.type for f in t.schema] == [f.type for f in c11.schema()]
    d = t.to_pydict()
    assert min(d["n_gt_boxes"]) > 0 and set(d["tb_subtype"]) <= set(c11.SUBTYPES)        # TB-with-box rows only
    c11.validate_table(path)


def test_mock_on_disk_is_current():
    """The committed-to-disk mock (make p3-mocks) must satisfy the current C11 rules."""
    p = REPO / "mock/artifacts/p3/xai/loc_radino_crop_m0.10_clahe0_bin_development_v1.parquet"
    if not p.exists():
        pytest.skip("mocks not generated (make p3-mocks)")
    c11.validate_table(p)
