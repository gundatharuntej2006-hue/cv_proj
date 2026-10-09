"""P3-01: pure-Python parts of notebooks/p3/p3_01_dfine_feasibility.py (no GPU, no D-FINE run)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "p3_01_dfine_feasibility", REPO / "notebooks" / "p3" / "p3_01_dfine_feasibility.py")
feas = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = feas
_spec.loader.exec_module(feas)


def _dfine_generate_scales(base_size, base_size_repeat):
    # Copied from third_party/D-FINE/src/data/dataloader.py @ 956d170 (keep in sync).
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@pytest.mark.parametrize("size, expected", [(512, 640), (640, 800)])
def test_max_train_scale_matches_dfine(size, expected):
    assert feas.max_train_scale(size) == expected == max(_dfine_generate_scales(size, 3))


def test_dfine_source_still_matches_copy():
    src = REPO / "third_party" / "D-FINE" / "src" / "data" / "dataloader.py"
    if not src.exists():
        pytest.skip("D-FINE submodule not checked out")
    text = src.read_text()
    assert "int(base_size * 1.25 / 32) * 32 - i * 32" in text


def test_write_config(tmp_path):
    dfine = tmp_path / "D-FINE"
    p = feas.write_config(tmp_path / "c.yml", dfine, "s", 640, tmp_path / "data", 8, 2,
                          amp=True, pretrained=False, run_dir=tmp_path / "run")
    cfg = json.loads(p.read_text())  # JSON is valid YAML for D-FINE's loader
    assert cfg["__include__"] == [str(dfine / "configs/dfine/custom/dfine_hgnetv2_s_custom.yml")]
    assert cfg["eval_spatial_size"] == [640, 640] and cfg["use_amp"] is True
    tl = cfg["train_dataloader"]
    assert tl["total_batch_size"] == 8 and tl["collate_fn"]["base_size"] == 640
    resize = [op for op in tl["dataset"]["transforms"]["ops"] if op["type"] == "Resize"]
    assert resize == [{"type": "Resize", "size": [640, 640]}]
    assert cfg["HGNetv2"] == {"pretrained": False}


def test_summarise_picks_chosen_probe():
    results = {"runs": [{
        "model": "s", "size": 512, "max_train_scale": 640, "chosen_batch": 8,
        "largest_workable_batch": 12, "s_per_epoch_mock": 10.0,
        "probes": [{"batch": 8, "ok": True, "peak_reserved_gb": 9.1},
                   {"batch": 12, "ok": True, "peak_reserved_gb": 14.6},
                   {"batch": 16, "ok": False, "oom": True}],
    }]}
    (row,) = feas.summarise(results)
    assert row["chosen_peak_reserved_gb"] == 9.1 and row["largest_workable_batch"] == 12
    assert list(row) == feas.SUMMARY_FIELDS


def test_smoke_args_are_cpu_and_tiny():
    a = feas.parse_args(["--smoke"])
    assert a.device == "cpu" and a.no_amp and a.no_pretrained and a.models == ["n"]
    assert a.sizes == [512, 640]
    full = feas.parse_args([])
    assert full.sizes == [512, 640] and not full.no_amp and full.device == "cuda"
