"""P3-16: latency E6 statistics (known numbers) and a smoke run on 5 fixture inputs.

P4's R4 run_batch is not in the repo yet: the smoke test injects a test-only stand-in that returns
the C14 stand-in fixtures with their timing_ms.
"""
from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "tests" / "p3_mock"):
    sys.path.insert(0, str(p))

from p3_case_results import P4_SCENARIOS  # noqa: E402

from p3_ops import latency  # noqa: E402


# --- statistics ------------------------------------------------------------------------------------

def test_median_p95_known_numbers():
    assert latency.median_p95(range(1, 101)) == (50.5, 95.05, 100)          # numpy linear interpolation
    assert latency.median_p95([10, 20, 30]) == (20.0, 29.0, 3)
    assert latency.median_p95([5.0, None, 7.0]) == (6.0, 6.9, 2)             # missing stage excluded
    m, p, n = latency.median_p95([])
    assert np.isnan(m) and np.isnan(p) and n == 0


def _rec(total, wall, status="ANALYSED", **stages):
    return {"case_id": "c_x", "status": status, "total": total, "wall_clock": wall,
            **{s: stages.get(s) for s in latency.STAGES if s != "total"}}


def test_verdict_uses_median_and_larger_of_total_and_wall():
    recs = [_rec(1500, 1600) for _ in range(10)] + [_rec(5000, 5100)]        # one slow outlier
    s = latency.summarise("gpu", recs, "machine", "abc1234", today="2026-10-09")
    assert s.verdict == "PASS" and s.decisive_ms == 1600.0                   # median, wall > total
    by = {r["stage"]: r for r in s.rows}
    assert by["total"]["p95_ms"] > 2000                                      # p95 shown, not decisive
    slow = latency.summarise("gpu", [_rec(1900, 2100) for _ in range(5)], "m", "x")
    assert slow.verdict == "FAIL"                                            # wall-clock median 2.1 s
    assert latency.summarise("cpu", [_rec(9000, 9100)] * 3, "m", "x").verdict == "PASS"
    assert latency.summarise("cpu", [_rec(10000, 9000)] * 3, "m", "x").verdict == "FAIL"   # < 10 s strictly


def test_rows_have_required_columns_and_counts():
    recs = [_rec(1000, 1100, gate=200, detect=None) for _ in range(4)] + [_rec(50, 60, "REJECTED_GATE", gate=50)]
    s = latency.summarise("gpu", recs, "CPU X; RAM 16 GB; GPU Y", "abc1234", today="2026-10-09")
    by = {r["stage"]: r for r in s.rows}
    assert [r["stage"] for r in s.rows] == [*latency.STAGES, "wall_clock"]
    assert all(set(r) == set(latency.SUMMARY_FIELDS) for r in s.rows)
    assert by["gate"]["n"] == 5 and by["detect"]["n"] == 0 and by["total"]["n"] == 5
    assert s.n_not_analysed == 1
    md = latency.table_markdown(s)
    assert "| Stage | Median (ms) | p95 (ms) | n |" in md and "PASS" in md and "1 not analysed" in md


# --- smoke on 5 fixture inputs ----------------------------------------------------------------------

def fake_run_batch():
    """Stand-in for p4_pipeline.api.run_batch: returns the C14 stand-ins in turn (with their timing)."""
    pool = list(P4_SCENARIOS.values())
    calls = []

    def run_batch(paths):
        calls.append(list(paths))
        return [copy.deepcopy(pool[(len(calls) - 1) % len(pool)]) for _ in paths]
    run_batch.calls = calls
    return run_batch


def test_smoke_5_fixture_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, "_force_mode", lambda mode: None)          # don't require a GPU here
    monkeypatch.setattr(latency, "machine_description", lambda mode: "CPU test; RAM 1.0 GB; GPU none")
    inputs = [tmp_path / f"img{i}.png" for i in range(5)]
    rb = fake_run_batch()
    s = latency.run("gpu", inputs=inputs, run_batch=rb, n_warmup=1, out_dir=tmp_path / "out")
    assert len(rb.calls) == 5 and all(len(c) == 1 for c in rb.calls)       # warm-up + one call per image
    assert s.n_images == 4
    with open(tmp_path / "out" / "e6_gpu.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["stage"] for r in rows] == [*latency.STAGES, "wall_clock"]
    assert rows[0]["mode"] == "gpu" and rows[0]["machine"].startswith("CPU test") and rows[0]["date"]
    per = (tmp_path / "out" / "e6_gpu_per_image.csv").read_text()
    assert "img" not in per and "png" not in per                             # case ids only, no paths
    assert "E6 latency, GPU" in (tmp_path / "out" / "e6_gpu.md").read_text()


def test_development_inputs_only_and_seeded(tmp_path):
    pytest.importorskip("pyarrow")
    from p3_det.data.fake_p1 import make_fake_p1_tree
    root = make_fake_p1_tree(tmp_path / "p1", n_per_split=12)
    a = latency.development_inputs(8, root=root)
    assert a == latency.development_inputs(8, root=root) and len(set(a)) == 8
    import csv as _csv
    with open(root / "data/splits/tbx11k_splits_v1.csv", newline="") as f:
        dev = {r["rel_path"] for r in _csv.DictReader(f) if r["split"] == "development"}
    assert all(p.relative_to(root / "data" / "raw" / "tbx11k").as_posix() in dev for p in a)
    with pytest.raises(RuntimeError, match="need"):
        latency.development_inputs(500, root=root)


def test_missing_r4_and_gpu_guard(monkeypatch):
    if __import__("importlib").util.find_spec("p4_pipeline") is None:
        with pytest.raises(RuntimeError, match="run_batch"):
            latency._p4_run_batch()
    import torch
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            latency._force_mode("gpu")
    with pytest.raises(RuntimeError, match="fresh process"):
        latency._force_mode("cpu")                                           # torch already imported here


def test_machine_description_and_memory_probe():
    d = latency.machine_description("cpu")
    assert d.startswith("CPU ") and "GPU none (CUDA disabled)" in d
    rss = latency.peak_rss_gb()
    assert rss is not None and 0 < rss < 1000                               # Windows (ctypes) and Unix probes
