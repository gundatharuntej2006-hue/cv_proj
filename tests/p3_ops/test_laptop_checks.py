"""F6 prep: laptop_checks report logic with a fake pytest runner (no GPU needed)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_ops import laptop_checks as lc  # noqa: E402

GPU = {"torch": "2.x", "cuda": True, "cuda_version": "12.x", "device": "NVIDIA GeForce RTX 4050 Laptop GPU",
       "total_gb": 6.44}


def _runner(vram_file: Path, peak: float, rc: int = 0, classifier: str = "stand-in RAD-DINO architecture"):
    def run(args):
        if "tests/p3_xai/test_gradcam.py" in args:
            time.sleep(0.01)
            vram_file.write_text(json.dumps({"peak_gb": peak, "peak_allocated_gb": peak - 0.3,
                                             "device": GPU["device"], "classifier": classifier}))
            return rc, "1 passed, 20 deselected in 9.0s"
        return 0, "R3 detect on GPU: median 120 ms, p95 150 ms (n=50)\n1 passed in 30s"
    return run


def test_no_checkpoint_gives_not_yet_and_report_is_written(tmp_path):
    vf = tmp_path / "vram.json"
    rc = lc.main(runner=_runner(vf, 3.2), out_dir=tmp_path, info=GPU, ckpt=lambda: (None, "D17 has no value yet"),
                 vram_file=vf)
    assert rc == 0
    (rep,) = tmp_path.glob("laptop_gpu_*.txt")
    text = rep.read_text(encoding="utf-8")
    assert "[PASS] Grad-CAM GPU (F6): peak reserved 3.2 GB" in text and "vs limit 5.5 GB" in text
    assert "NOT_YET: no trained checkpoint" in text and "architecture-level evidence only" in text


def test_vram_over_limit_fails(tmp_path):
    vf = tmp_path / "vram.json"
    assert lc.main(runner=_runner(vf, 5.6, rc=1), out_dir=tmp_path, info=GPU, ckpt=lambda: (None, "x"),
                   vram_file=vf) == 1
    assert "[FAIL] Grad-CAM GPU" in next(tmp_path.glob("laptop_gpu_*.txt")).read_text(encoding="utf-8")


def test_latency_runs_only_with_a_checkpoint(tmp_path):
    vf = tmp_path / "vram.json"
    ck = tmp_path / "ckpt_best.pt"
    ck.write_bytes(b"x")
    assert lc.main(runner=_runner(vf, 3.0, classifier="p2_cls.api.classifier_module()"), out_dir=tmp_path, info=GPU,
                   ckpt=lambda: (ck, "D17"), vram_file=vf) == 0
    text = next(tmp_path.glob("laptop_gpu_*.txt")).read_text(encoding="utf-8")
    assert "[PASS] detect() latency" in text and "median 120 ms" in text
    assert "architecture-level" not in text                                  # real classifier


def test_no_cuda_is_reported_not_hidden(tmp_path):
    assert lc.main(runner=lambda a: (_ for _ in ()).throw(AssertionError("must not run")), out_dir=tmp_path,
                   info={"torch": "2.x", "cuda": False}) == 1
    assert "no CUDA GPU" in next(tmp_path.glob("laptop_gpu_*.txt")).read_text(encoding="utf-8")


def test_stale_vram_file_is_not_reused(tmp_path):
    vf = tmp_path / "vram.json"
    vf.write_text(json.dumps({"peak_gb": 1.0}))
    old = time.time() - 3600
    import os
    os.utime(vf, (old, old))
    res = lc.check_gradcam(lambda a: (5, "no tests ran"), vf)
    assert res["status"] == "NOT_RUN" and "no fresh VRAM record" in res["detail"]
