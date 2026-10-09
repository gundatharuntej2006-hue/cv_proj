"""P3-08: real R3 detect() and the C10 export.

CPU tests use a tiny real D-FINE-N checkpoint written in P3-02 format plus decision files in a temp
dir, so the whole real path runs (decision reading, SHA check, EMA weights, crop/resize, inference,
mapping back, C10). The latency test (marked gpu) uses the REAL locked D16-D18 and checkpoint.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import pytest

for mod in ("torch", "yaml", "scipy", "faster_coco_eval", "calflops", "pyarrow"):
    pytest.importorskip(mod)
REPO = Path(__file__).resolve().parents[2]
if not (REPO / "third_party" / "D-FINE" / "train.py").exists():
    pytest.skip("D-FINE submodule not checked out", allow_module_level=True)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "p3_mock"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from p3_det import api, c10, export_c10  # noqa: E402
from p3_det.data import DataPaths  # noqa: E402
from p3_det.data.fake_p1 import make_fake_p1_tree  # noqa: E402
from p3_det.train import dfine_bridge as bridge  # noqa: E402
from p3_det.train.config import RunConfig, load_run_config  # noqa: E402
from p3_ops.cloud import save_checkpoint  # noqa: E402

warnings.filterwarnings("ignore")
THR = 0.3
RUN_DIR = "dfine_s_512_m0.10_clahe0"


def _tiny_config() -> dict:
    raw = load_run_config("dfine_512.yaml").raw
    raw["model"].update(variant="n", dfine_config="configs/dfine/custom/dfine_hgnetv2_n_custom.yml", init="none")
    return raw


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p3_08")
    data = make_fake_p1_tree(tmp / "p1", n_per_split=60)
    raw = _tiny_config()
    torch.manual_seed(0)
    c = bridge.build(RunConfig(raw, Path("x.yaml")), "cpu", amp=False)
    with torch.no_grad():                         # untrained heads score ~0.01; lift them so boxes exist
        for name, p in c.model.named_parameters():
            if "score_head" in name and name.endswith("bias"):
                p += 3.0
    c.ema.module.load_state_dict(c.model.state_dict())
    ck = tmp / "ckpt"
    save_checkpoint(c.model, c.optimizer, c.lr_scheduler, 3, {"dev_ap50": 0.1}, 0, ck, is_best=True,
                    extra={"ema": c.ema})
    ckpt = ck / "ckpt_best.pt"
    import hashlib
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    runs = tmp / "runs"
    (runs / RUN_DIR).mkdir(parents=True)
    prep = {"margin": 0.1, "clahe": False, "provisional": True}
    (runs / RUN_DIR / "run_meta.json").write_text(json.dumps(
        {"run_name": "dfine_n_512", "run_dir": RUN_DIR, "input_size": 512, "config": raw, "preprocessing": prep}))
    dd = tmp / "decisions"
    dd.mkdir()
    for name, value in (("p3_det_input_size.json", {"input_size": 512}),
                        ("p3_det_checkpoint.json", {"input_size": 512, "run_dir": RUN_DIR, "epoch": 3,
                                                    "checkpoint": {"sha256": sha}}),
                        ("p3_det_conf_threshold.json", {"conf_threshold": THR})):
        (dd / name).write_text(json.dumps({"decision_id": name, "value": value, "locked": False}))
    env = {"TB_DECISIONS_DIR": str(dd), "TB_DET_RUNS_DIR": str(runs), "TB_DET_CHECKPOINT": str(ckpt),
           "TB_DET_ALLOW_UNLOCKED": "1", "TB_DET_DEVICE": "cpu"}
    old = {k: os.environ.get(k) for k in [*env, "TB_MOCK"]}
    os.environ.update(env)
    os.environ.pop("TB_MOCK", None)
    api.reset_cache()
    yield {"tmp": tmp, "data": data, "dd": dd, "ckpt": ckpt, "sha": sha, "out": tmp / "out"}
    api.reset_cache()
    for k, v in old.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@pytest.fixture(scope="module")
def c10_dev(setup):
    return export_c10.export("development", paths=DataPaths(setup["data"]), out_dir=setup["out"],
                             decisions_dir=setup["dd"])


def test_c10_validates(c10_dev, setup):
    import standin_contracts as sc
    c10.validate_c10(c10_dev)
    doc = sc.validate_c10(c10_dev)                                # P3-03 stand-in validator as well
    assert c10_dev.name == "detections_tbx11k_development_v1.json"
    assert doc["input_size"] == 512 and doc["conf_threshold"] == THR and doc["coords"] == "full512"
    assert doc["model_id"] == f"dfine_n_512_e003_{setup['sha'][:8]}"
    assert len(doc["images"]) == 60                               # every readable development image
    assert sum(len(im["boxes"]) for im in doc["images"]) > 0


def _images(setup, doc, n=50):
    import pyarrow.parquet as pq
    man = {r["image_id"]: r for r in pq.read_table(setup["data"] / "data/processed/v1/manifest.parquet").to_pylist()}
    for im in doc["images"][:n]:
        yield im, np.asarray(Image.open(setup["data"] / man[im["image_id"]]["path_img512"]).convert("L"))


def test_detect_equals_c10_for_50_development_images(c10_dev, setup):
    doc = json.loads(c10_dev.read_text())
    checked = 0
    for im, img in _images(setup, doc, 50):
        r = api.detect(img, im["crop_box"])
        file_boxes = np.array([b[:4] for b in im["boxes"]]).reshape(-1, 4)
        file_scores = np.array([b[4] for b in im["boxes"]])
        assert r.boxes_full512.shape == file_boxes.shape
        assert np.allclose(r.boxes_full512, file_boxes, atol=1e-3)
        assert np.array_equal(r.scores.astype(float), file_scores)
        assert r.n_lesions_above_thr == im["n_lesions_above_thr"]
        checked += 1
    assert checked == 50


def test_boxes_within_512_and_inside_crop(c10_dev, setup):
    doc = json.loads(c10_dev.read_text())
    for im, img in _images(setup, doc, 20):
        r = api.detect(img, im["crop_box"])
        x0, y0, x1, y1 = im["crop_box"]
        b = r.boxes_full512
        assert ((b >= 0) & (b <= 512)).all()
        assert (b[:, 0] >= x0).all() and (b[:, 2] <= x1).all() and (b[:, 1] >= y0).all() and (b[:, 3] <= y1).all()
        assert (r.scores >= 0.01).all() and list(r.scores) == sorted(r.scores, reverse=True)
        assert r.boxes_full512.dtype == np.float32 and isinstance(r.n_lesions_above_thr, int)


def test_model_loaded_once(setup, monkeypatch):
    api.reset_cache()
    calls = []
    real = api._Detector.__init__

    def counting(self, s):
        calls.append(1)
        real(self, s)
    monkeypatch.setattr(api._Detector, "__init__", counting)
    img = np.full((512, 512), 90, np.uint8)
    for _ in range(3):
        api.detect(img, [10, 10, 500, 500])
    assert len(calls) == 1


def test_rgb_input_matches_gray(setup):
    img = (np.arange(512 * 512) % 200).astype(np.uint8).reshape(512, 512)
    a = api.detect(img, [20, 0, 490, 500])
    b = api.detect(np.stack([img] * 3, -1), [20, 0, 490, 500])
    assert np.array_equal(a.scores, b.scores)
    with pytest.raises(ValueError, match="uint8"):
        api.detect(img.astype(np.float32), [20, 0, 490, 500])


def test_refuses_unlocked_and_wrong_checkpoint(setup, monkeypatch, tmp_path):
    monkeypatch.delenv("TB_DET_ALLOW_UNLOCKED")
    api.reset_cache()
    with pytest.raises(RuntimeError, match="not locked"):
        api.detect(np.zeros((512, 512), np.uint8), [0, 0, 512, 512])
    monkeypatch.setenv("TB_DET_ALLOW_UNLOCKED", "1")
    bad = tmp_path / "ckpt_best.pt"
    bad.write_bytes(setup["ckpt"].read_bytes() + b"x")
    monkeypatch.setenv("TB_DET_CHECKPOINT", str(bad))
    api.reset_cache()
    with pytest.raises(RuntimeError, match="SHA-256"):
        api.detect(np.zeros((512, 512), np.uint8), [0, 0, 512, 512])
    monkeypatch.setenv("TB_DET_CHECKPOINT", str(setup["ckpt"]))
    api.reset_cache()


@pytest.mark.parametrize("split", ["internal_test", "challenge_test", "external_test", "train"])
def test_export_refuses_other_splits(setup, split):
    with pytest.raises(PermissionError):
        export_c10.export(split, paths=DataPaths(setup["data"]), out_dir=setup["out"], decisions_dir=setup["dd"])


def test_export_refuses_preprocessing_mismatch(setup, tmp_path):
    dd = tmp_path / "dec"
    dd.mkdir()
    (dd / "p2_crop_margin.json").write_text(json.dumps({"value": {"margin": 0.05}, "locked": True}))
    (dd / "p2_clahe.json").write_text(json.dumps({"value": {"clahe": False}, "locked": True}))
    with pytest.raises(RuntimeError, match="trained with margin"):
        export_c10.export("development", paths=DataPaths(setup["data"]), out_dir=tmp_path, decisions_dir=dd)


def test_mock_mode_still_serves_stub(setup, monkeypatch):
    monkeypatch.setenv("TB_MOCK", "1")
    r = api.detect(np.zeros((512, 512), np.uint8), [10, 0, 507, 493])
    assert isinstance(r, api.DetResult) and api.detector_info()["mock"] is True


# --- latency on the laptop GPU (real locked model) ---------------------------------------------------

@pytest.mark.gpu
def test_latency_rtx4050_under_500ms():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    for k in ("TB_DECISIONS_DIR", "TB_DET_RUNS_DIR", "TB_DET_CHECKPOINT", "TB_DET_ALLOW_UNLOCKED", "TB_DET_DEVICE",
              "TB_MOCK"):
        os.environ.pop(k, None)
    api.reset_cache()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (512, 512), dtype=np.uint8)
    crop = [40, 20, 470, 490]
    for _ in range(5):                                            # warm-up (cuDNN autotune, allocator)
        api.detect(img, crop)
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        api.detect(img, crop)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    med, p95 = statistics.median(times), float(np.percentile(times, 95))
    print(f"\nR3 detect on {torch.cuda.get_device_name(0)}: median {med:.0f} ms, p95 {p95:.0f} ms (n=50)")
    assert med <= 500, f"median {med:.0f} ms > 500 ms"
