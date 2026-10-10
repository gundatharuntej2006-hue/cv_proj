"""F6: GPU checks on the laptop (RTX 4050), one command, one text report.

    python src/p3_ops/laptop_checks.py          (from the cv_proj folder; Windows, Linux and macOS alike)

1. Grad-CAM on the GPU: runs tests/p3_xai/test_gradcam.py -m gpu (one explain() call after a warm-up) and
   records peak reserved VRAM against the 5.5 GB limit (D19). Uses P2's classifier_module() when it is
   installed, otherwise the stand-in RAD-DINO architecture (same size, random weights); the report says which.
2. detect() latency (<= 500 ms/image): runs tests/p3_det/test_r3_detect.py -m gpu ONLY if a trained checkpoint
   exists (D17 filled and its ckpt_best.pt present, or TB_DET_CHECKPOINT set); otherwise the report says
   "NOT_YET: no trained checkpoint".

Report: artifacts/p3/verify/laptop_gpu_<date>.txt. Exit code 0 = every check that ran passed.
No data from any split is read: both tests use synthetic images.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "p3" / "verify"
VRAM_FILE = REPO / "artifacts" / "p3" / "xai" / "vram_explain.json"
VRAM_LIMIT_GB = 5.5
LATENCY_LIMIT_MS = 500


def run_pytest(args: list[str]) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")]),
           "PYTHONIOENCODING": "utf-8"}
    env.pop("TB_MOCK", None)
    p = subprocess.run([sys.executable, "-m", "pytest", *args, "-p", "no:cacheprovider", "-rA"], cwd=REPO, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda": False}
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "cuda_version": torch.version.cuda}
    if info["cuda"]:
        prop = torch.cuda.get_device_properties(0)
        info.update(device=prop.name, total_gb=round(prop.total_memory / 1e9, 2))
    return info


def trained_checkpoint() -> tuple[Path | None, str]:
    """(checkpoint path, why) - a checkpoint counts only if it exists AND D17 names it (or TB_DET_CHECKPOINT)."""
    env = os.environ.get("TB_DET_CHECKPOINT")
    if env:
        return (Path(env), "TB_DET_CHECKPOINT") if Path(env).exists() else (None, f"TB_DET_CHECKPOINT={env} missing")
    d17 = REPO / "decisions" / "p3_det_checkpoint.json"
    v = json.loads(d17.read_text()).get("value") if d17.exists() else None
    if not v:
        return None, "D17 has no value yet (no training run selected)"
    p = REPO / "artifacts" / "p3" / "det" / "checkpoints" / v["run_dir"] / "ckpt_best.pt"
    return (p, "D17") if p.exists() else (None, f"{p.relative_to(REPO)} not downloaded from Drive")


def _summary(out: str) -> str:
    lines = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed|skipped|error|deselected)", l)]
    return lines[-1].strip("= ") if lines else "(no pytest summary)"


def check_gradcam(runner: Callable = run_pytest, vram_file: Path = VRAM_FILE) -> dict:
    t0 = time.time()
    rc, out = runner(["tests/p3_xai/test_gradcam.py", "-m", "gpu", "-s", "-q", "-k", "vram"])
    rec = json.loads(vram_file.read_text()) if vram_file.exists() and vram_file.stat().st_mtime >= t0 else None
    if rec is None:
        status = "FAIL" if rc not in (0, 5) or "failed" in _summary(out) else "NOT_RUN"
        return {"name": "Grad-CAM GPU (F6)", "status": status, "detail": f"no fresh VRAM record; pytest: {_summary(out)}",
                "log": out[-3000:]}
    ok = rec["peak_gb"] <= VRAM_LIMIT_GB and rc == 0
    real = "stand-in" not in rec.get("classifier", "")
    return {"name": "Grad-CAM GPU (F6)", "status": "PASS" if ok else "FAIL",
            "detail": f"peak reserved {rec['peak_gb']} GB (allocated {rec.get('peak_allocated_gb')} GB) vs limit "
                      f"{VRAM_LIMIT_GB} GB on {rec.get('device')}; classifier: {rec.get('classifier')}"
                      + ("" if real else " -> architecture-level evidence only; repeat once P2's classifier exists"),
            "log": out[-3000:]}


def check_detect_latency(runner: Callable = run_pytest, ckpt: Callable = trained_checkpoint) -> dict:
    path, why = ckpt()
    if path is None:
        return {"name": "detect() latency", "status": "NOT_YET", "detail": f"NOT_YET: no trained checkpoint ({why})"}
    rc, out = runner(["tests/p3_det/test_r3_detect.py", "-m", "gpu", "-s", "-q", "-k", "latency"])
    lat = re.findall(r"[Mm]edian[^\n]*?([\d.]+)\s*ms", out)
    return {"name": "detect() latency", "status": "PASS" if rc == 0 else "FAIL",
            "detail": f"checkpoint {path.name} ({why}); target <= {LATENCY_LIMIT_MS} ms/image; "
                      f"median {lat[-1] + ' ms' if lat else 'see log'}; pytest: {_summary(out)}", "log": out[-3000:]}


def report_text(info: dict, checks: list[dict], when: datetime) -> str:
    L = ["P3 laptop GPU checks (F6)", f"date: {when.isoformat(timespec='seconds')}",
         f"machine: {platform.platform()} | python {platform.python_version()} | torch {info.get('torch')} "
         f"(CUDA {info.get('cuda_version')})",
         f"GPU: {info.get('device', 'none')} ({info.get('total_gb', '-')} GB)", "",
         "Synthetic images only; no development, calibration or sealed data was read.", ""]
    for c in checks:
        L += [f"[{c['status']}] {c['name']}: {c['detail']}"]
    L += ["", "--- pytest output (tail) ---"]
    for c in checks:
        if c.get("log"):
            L += [f"## {c['name']}", c["log"].rstrip(), ""]
    return "\n".join(L) + "\n"


def main(argv=None, *, runner: Callable = run_pytest, out_dir: Path = OUT_DIR, info: dict | None = None,
         ckpt: Callable = trained_checkpoint, vram_file: Path = VRAM_FILE) -> int:
    info = gpu_info() if info is None else info
    when = datetime.now()
    if not info.get("cuda"):
        checks = [{"name": "Grad-CAM GPU (F6)", "status": "FAIL",
                   "detail": "no CUDA GPU visible to torch (install the CUDA build of torch, check the NVIDIA driver)"},
                  {"name": "detect() latency", "status": "NOT_RUN", "detail": "no CUDA GPU"}]
    else:
        checks = [check_gradcam(runner, vram_file), check_detect_latency(runner, ckpt)]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"laptop_gpu_{when.date().isoformat()}.txt"
    out.write_text(report_text(info, checks, when), encoding="utf-8")
    for c in checks:
        print(f"[{c['status']}] {c['name']}: {c['detail']}")
    print(f"report: {out}")
    return 1 if any(c["status"] == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
