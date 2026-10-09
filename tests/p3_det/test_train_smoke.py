"""P3-06: end-to-end CPU smoke of the D-FINE training wrapper on FAKE P1 data.

Trains D-FINE-N from scratch for 2 tiny epochs at 512 and 640 (with a simulated stop + resume),
then runs select_size on the two runs. Skipped when D-FINE's dependencies are not installed
(e.g. in CI, which installs only requirements/base.txt).
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import pytest

for mod in ("torch", "yaml", "scipy", "faster_coco_eval", "calflops", "pyarrow"):
    pytest.importorskip(mod)
REPO = Path(__file__).resolve().parents[2]
if not (REPO / "third_party" / "D-FINE" / "train.py").exists():
    pytest.skip("D-FINE submodule not checked out", allow_module_level=True)
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from p3_det import select_size as ss  # noqa: E402
from p3_det.data.fake_p1 import make_fake_p1_tree  # noqa: E402
from p3_det.train import run as tr  # noqa: E402
from p3_det.train.config import load_run_config  # noqa: E402

warnings.filterwarnings("ignore")


def _smoke_cfg(tmp: Path, name: str) -> Path:
    raw = load_run_config(name).raw
    raw["model"].update(variant="n", dfine_config="configs/dfine/custom/dfine_hgnetv2_n_custom.yml", init="none")
    raw["schedule"].update(epochs=2, batch_size=2, eval_batch_size=4, num_workers=0, amp=False,
                           multiscale_stop_epoch=1)
    p = tmp / name
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p3_06")
    data = make_fake_p1_tree(tmp / "p1", n_per_split=6)
    out = {}
    for name in ("dfine_512.yaml", "dfine_640.yaml"):
        cfg = _smoke_cfg(tmp, name)
        kw = dict(device="cpu", data_root=data, ckpt_root=tmp / "ckpt", decisions_dir=tmp / "no_decisions")
        tr.train(cfg, max_epochs=1, **kw)        # "session killed" after epoch 0
        out[name] = tr.train(cfg, **kw)          # resumes at epoch 1
    return out


def test_run_folder_contents(runs):
    for rd in runs.values():
        assert rd.name.endswith("_m0.10_clahe0_PROVISIONAL")
        for f in ("run_meta.json", "train_log.csv", "dev_ground_truth.json", "dev_detections_last.json",
                  "dev_detections_best.json", "ckpt_best.pt", "best.json"):
            assert (rd / f).exists(), f
        meta = json.loads((rd / "run_meta.json").read_text())
        assert meta["seed"] == 0 and meta["seed_util"] and "resumed_at" in meta
        assert meta["splits"] == {"train": "train", "eval": "development"}
        assert meta["preprocessing"]["provisional"] is True


def test_log_has_each_epoch_once_after_resume(runs):
    for rd in runs.values():
        with open(rd / "train_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [int(r["epoch"]) for r in rows] == [0, 1]
        assert all(r["dev_ap50"] != "" and int(r["dev_n_images"]) > 0 for r in rows)


def test_detections_in_full512_inside_crop(runs):
    for rd in runs.values():
        det = json.loads((rd / "dev_detections_last.json").read_text())
        assert det["coords"] == "full512" and det["epoch"] == 1
        for im in det["images"]:
            for x0, y0, x1, y1, s in im["boxes"]:
                assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512 and s >= 0.01


def test_select_size_on_smoke_runs(runs, tmp_path):
    path = tmp_path / "p3_det_input_size.json"
    ss.write_criterion(path)
    doc, code = ss.compare(runs["dfine_512.yaml"], runs["dfine_640.yaml"], path=path,
                           evidence_dir=tmp_path / "ev", allow_uncommitted=True)
    # random-init 2-epoch models are useless: the contingency must fire
    assert code == 3 and doc["value"]["contingency"]["triggered"] and not doc["locked"]
    assert doc["value"]["preprocessing"]["provisional"] is True
