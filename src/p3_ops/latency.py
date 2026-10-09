"""P3-16: latency experiment E6 (full pipeline, P4's R4 run_batch) on development images.

    python -m p3_ops.latency --mode gpu       # RTX 4050
    python -m p3_ops.latency --mode cpu       # CUDA forced off (set before torch is imported)

Protocol (fixed before measuring; nothing is tuned to pass):
* Images: development split only (C01), readable (C04 decode_status ok), seeded sample (seed 0) of
  3 + 100; the input given to run_batch is the original file (data/raw/tbx11k/<rel_path>), as at deployment.
* Warm-up: the first 3 images are processed and discarded.
* Each of the 100 images is processed with its own run_batch([path]) call so that the wall-clock time
  of every image is measured independently (cross-check of CaseResult.timing_ms.total).
* Per stage (standardise, gate, segment, classify, detect, explain, total) and for wall-clock: median and
  p95 (numpy linear interpolation) and n. Stages missing from a CaseResult (rejected / unreadable
  images) are excluded from that stage and counted.
* Verdict: PASS if the MEDIAN per-image time is below the target (GPU < 2000 ms, CPU < 10000 ms), using
  the larger of the pipeline total and the wall-clock median; p95 is shown alongside.
* CPU mode also records the process's peak memory and flags it if above 8 GB.

Outputs (artifacts/p3/latency/): e6_<mode>.csv (summary), e6_<mode>_per_image.csv (case_id + timings,
no paths), e6_<mode>.md (table to paste into the report).
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "p3" / "latency"
STAGES = ("standardise", "gate", "segment", "classify", "detect", "explain", "total")
TARGET_MS = {"gpu": 2000.0, "cpu": 10000.0}
N_WARMUP, N_MEASURE, SEED = 3, 100, 0
RAM_BUDGET_GB = 8.0
SUMMARY_FIELDS = ["mode", "stage", "median_ms", "p95_ms", "n", "machine", "date", "git_sha"]


# --- statistics ------------------------------------------------------------------------------------

def median_p95(values: Sequence[float]) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return float("nan"), float("nan"), 0
    return float(np.median(v)), float(np.percentile(v, 95)), len(v)


@dataclass
class Summary:
    mode: str
    rows: list[dict]
    verdict: str
    decisive_ms: float
    target_ms: float
    n_images: int
    n_not_analysed: int
    peak_rss_gb: float | None
    machine: str


def summarise(mode: str, per_image: list[dict], machine: str, git_sha: str, today: str | None = None,
              peak_rss_gb: float | None = None) -> Summary:
    today = today or date.today().isoformat()
    rows = []
    for stage in (*STAGES, "wall_clock"):
        med, p95, n = median_p95([r.get(stage) for r in per_image])
        rows.append({"mode": mode, "stage": stage, "median_ms": round(med, 1), "p95_ms": round(p95, 1), "n": n,
                     "machine": machine, "date": today, "git_sha": git_sha})
    by = {r["stage"]: r for r in rows}
    decisive = max(by["total"]["median_ms"], by["wall_clock"]["median_ms"])
    target = TARGET_MS[mode]
    verdict = "PASS" if decisive < target else "FAIL"
    return Summary(mode, rows, verdict, decisive, target, len(per_image),
                   sum(r.get("status") != "ANALYSED" for r in per_image), peak_rss_gb, machine)


def table_markdown(s: Summary) -> str:
    lines = [f"**E6 latency, {s.mode.upper()}** - {s.machine}",
             "",
             "| Stage | Median (ms) | p95 (ms) | n |", "|---|---:|---:|---:|"]
    for r in s.rows:
        name = "wall-clock (cross-check)" if r["stage"] == "wall_clock" else r["stage"]
        lines.append(f"| {name} | {r['median_ms']:.0f} | {r['p95_ms']:.0f} | {r['n']} |")
    p95_total = next(r["p95_ms"] for r in s.rows if r["stage"] == "total")
    lines += ["", f"Target < {s.target_ms / 1000:.0f} s per image (median): **{s.verdict}** "
                  f"(median {s.decisive_ms / 1000:.2f} s, larger of pipeline total and wall-clock; "
                  f"p95 total {p95_total / 1000:.2f} s). {s.n_images} development images after "
                  f"{N_WARMUP} warm-up images; {s.n_not_analysed} not analysed (rejected/unreadable)."]
    if s.peak_rss_gb is not None:
        flag = "" if s.peak_rss_gb <= RAM_BUDGET_GB else f" - ABOVE the {RAM_BUDGET_GB:.0f} GB budget"
        lines.append(f"Peak process memory: {s.peak_rss_gb:.2f} GB{flag}.")
    return "\n".join(lines)


# --- machine ---------------------------------------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m.ullTotalPhys / 1e9
        return None
    try:
        kb = next(int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal"))
        return kb / 1e6
    except (OSError, StopIteration):
        return None


def peak_rss_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE                     # 64-bit pseudo-handle
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 1e9
        return None
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)


def machine_description(mode: str) -> str:
    ram = _ram_gb()
    gpu = "none (CUDA disabled)"
    if mode == "gpu":
        import torch
        gpu = torch.cuda.get_device_name(0)
    return f"CPU {_cpu_name()}; RAM {ram:.1f} GB; GPU {gpu}; {platform.system()} {platform.release()}" \
        if ram else f"CPU {_cpu_name()}; GPU {gpu}; {platform.system()} {platform.release()}"


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --- inputs + run -------------------------------------------------------------------------------------

def development_inputs(n: int, seed: int = SEED, root: Path | None = None) -> list[Path]:
    """Seeded sample of readable development images, as original input files."""
    import pyarrow.parquet as pq

    from p3_det.data.config import DataPaths, check_split
    check_split("development")
    paths = DataPaths(root) if root else DataPaths.default()
    with open(paths.splits, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "development"]
    ok = {r["image_id"] for r in pq.read_table(paths.manifest).to_pylist() if r["decode_status"] == "ok"}
    rows = sorted((r for r in rows if r["image_id"] in ok), key=lambda r: r["image_id"])
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} readable development images, need {n}")
    pick = np.random.default_rng(seed).choice(len(rows), size=n, replace=False)
    return [paths.root / "data" / "raw" / "tbx11k" / rows[i]["rel_path"] for i in pick]


def measure(inputs: Sequence[Path], run_batch: Callable, n_warmup: int = N_WARMUP) -> list[dict]:
    """Warm-up (discarded), then one run_batch([path]) per image; returns per-image records."""
    for p in inputs[:n_warmup]:
        run_batch([str(p)])
    out = []
    for p in inputs[n_warmup:]:
        t0 = time.perf_counter()
        results = run_batch([str(p)])
        wall = (time.perf_counter() - t0) * 1000
        if len(results) != 1:
            raise RuntimeError(f"run_batch returned {len(results)} results for one input")
        cr = results[0]
        cr = cr if isinstance(cr, dict) else getattr(cr, "__dict__", {})
        t = cr.get("timing_ms") or {}
        out.append({"case_id": cr.get("case_id"), "status": cr.get("status"), "wall_clock": round(wall, 1),
                    **{s: t.get(s) for s in STAGES}})
    return out


def _p4_run_batch() -> Callable:
    try:
        from p4_pipeline.api import run_batch
    except ImportError as e:
        raise RuntimeError("p4_pipeline.api.run_batch (R4, P4) is not available") from e
    return run_batch


def _force_mode(mode: str) -> None:
    if mode == "cpu":
        if "torch" in sys.modules:
            raise RuntimeError("CPU mode must disable CUDA before torch is imported; run in a fresh process")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TB_DET_DEVICE"] = os.environ["TB_XAI_DEVICE"] = "cpu"
    import torch
    if mode == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU mode requested but CUDA is not available (not falling back to CPU)")
    if mode == "cpu" and torch.cuda.is_available():
        raise RuntimeError("CUDA still visible in CPU mode")


def write_outputs(s: Summary, per_image: list[dict], out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"summary": out_dir / f"e6_{s.mode}.csv", "per_image": out_dir / f"e6_{s.mode}_per_image.csv",
             "markdown": out_dir / f"e6_{s.mode}.md"}
    with open(paths["summary"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(s.rows)
    with open(paths["per_image"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "status", *STAGES, "wall_clock"])
        w.writeheader()
        w.writerows(per_image)
    paths["markdown"].write_text(table_markdown(s) + "\n", encoding="utf-8")
    return paths


def run(mode: str, *, inputs: Sequence[Path] | None = None, run_batch: Callable | None = None,
        n_warmup: int = N_WARMUP, out_dir: Path = OUT_DIR) -> Summary:
    if mode not in TARGET_MS:
        raise ValueError("mode must be 'gpu' or 'cpu'")
    _force_mode(mode)
    inputs = list(inputs) if inputs is not None else development_inputs(N_WARMUP + N_MEASURE)
    per_image = measure(inputs, run_batch or _p4_run_batch(), n_warmup)
    s = summarise(mode, per_image, machine_description(mode), git_sha(),
                  peak_rss_gb=peak_rss_gb() if mode == "cpu" else None)
    write_outputs(s, per_image, out_dir)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(TARGET_MS), required=True)
    a = ap.parse_args(argv)
    s = run(a.mode)
    print(table_markdown(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
