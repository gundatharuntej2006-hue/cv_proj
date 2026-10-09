"""P3-01: D-FINE T4 feasibility on synthetic COCO (input sizes 512 and 640, AMP).

For each model variant x input size this measures, on the GPU it runs on:
  * peak GPU memory per batch size at the LARGEST multi-scale training size
    (D-FINE trains at up to 1.25x the base size: 512 -> 640 px, 640 -> 800 px),
  * the largest workable batch (no OOM) and the chosen batch (largest with
    peak reserved memory <= 14 GB, decimal GB = 1e9 bytes),
  * seconds per epoch with the chosen batch (1 warm-up epoch, then timed),
    plus the peak memory seen during that real multi-scale epoch.

Synthetic data only (src/p3_det/synthetic_coco.py): no real images, no sealed data.

Run on Kaggle/Colab T4 (one GPU):
    python notebooks/p3/p3_01_dfine_feasibility.py --models s m --n-train 6600
CPU smoke test (checks the plumbing, no memory numbers):
    python notebooks/p3/p3_01_dfine_feasibility.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_det.synthetic_coco import make_synthetic_coco  # noqa: E402

GB = 1e9
PEAK_LIMIT_GB = 14.0
DEFAULT_BATCHES = (2, 4, 6, 8, 12, 16, 20, 24, 32, 48)
# C06 uses category_id = 1 and D-FINE (remap_mscoco_category=False) uses the id as the label
# index, so labels must lie in [0, num_classes). P3-04 decides between remapping 1 -> 0 with
# num_classes=1 and keeping num_classes=2; the memory difference is negligible here.
NUM_CLASSES = 2


def max_train_scale(base: int) -> int:
    """Largest multi-scale size, same formula as D-FINE generate_scales()."""
    return int(base * 1.25 / 32) * 32


def set_seed(seed: int) -> str:
    try:
        from tbcore.seed import set_all  # P4 utility, preferred
        set_all(seed)
        return "tbcore.seed.set_all"
    except ImportError:
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return "fallback (tbcore.seed not available)"


def write_config(path: Path, dfine_dir: Path, model: str, size: int, data_dir: Path,
                 batch: int, workers: int, amp: bool, pretrained: bool, run_dir: Path) -> Path:
    """Write a D-FINE config (JSON is valid YAML) that overrides the official custom config."""
    ops = [
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [size, size]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]
    val_ops = [{"type": "Resize", "size": [size, size]},
               {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]
    data = {"img_folder": str(data_dir / "train"), "ann_file": str(data_dir / "train.json")}
    cfg = {
        "__include__": [str(dfine_dir / f"configs/dfine/custom/dfine_hgnetv2_{model}_custom.yml")],
        "output_dir": str(run_dir),
        "num_classes": NUM_CLASSES,
        "remap_mscoco_category": False,
        "eval_spatial_size": [size, size],
        "use_amp": amp,
        "print_freq": 10**6,
        "HGNetv2": {"pretrained": pretrained},
        "train_dataloader": {
            "total_batch_size": batch, "num_workers": workers,
            "dataset": {**data, "transforms": {"ops": ops}},
            "collate_fn": {"base_size": size},
        },
        "val_dataloader": {
            "total_batch_size": batch, "num_workers": workers,
            "dataset": {**data, "transforms": {"ops": val_ops}},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1))
    return path


def build_solver(cfg_path: Path, device: str, seed: int, amp: bool):
    from src.core import YAMLConfig
    from src.solver import TASKS
    cfg = YAMLConfig(str(cfg_path), device=device, seed=seed, use_amp=amp)
    solver = TASKS["detection"](cfg)
    solver.train()
    return solver


def _train_step(solver, samples, targets, step: int) -> None:
    """One optimisation step, same as D-FINE det_engine.train_one_epoch."""
    import torch
    model, criterion, opt, scaler = solver.model, solver.criterion, solver.optimizer, solver.scaler
    metas = dict(epoch=0, step=step, global_step=step, epoch_step=1000)
    if scaler is not None:
        with torch.autocast(device_type=str(solver.device), cache_enabled=True):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type=str(solver.device), enabled=False):
            loss = sum(criterion(outputs, targets, **metas).values())
        scaler.scale(loss).backward()
        if solver.cfg.clip_max_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), solver.cfg.clip_max_norm)
        scaler.step(opt)
        scaler.update()
    else:
        outputs = model(samples, targets=targets)
        loss = sum(criterion(outputs, targets, **metas).values())
        loss.backward()
        opt.step()
    opt.zero_grad()
    if solver.ema is not None:
        solver.ema.update(model)


def _cuda_peaks() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"peak_alloc_gb": None, "peak_reserved_gb": None}
    torch.cuda.synchronize()
    return {"peak_alloc_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3)}


def _reset_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def probe_batch(solver, batch: int, scale: int, steps: int = 3) -> dict:
    """Train ``steps`` steps at the worst-case multi-scale size; report peaks or OOM."""
    import torch
    import torch.nn.functional as F
    ds = solver.train_dataloader.dataset
    dev = solver.device
    solver.model.train()
    solver.criterion.train()
    _reset_cuda()
    try:
        for step in range(steps):
            items = [ds[(step * batch + j) % len(ds)] for j in range(batch)]
            samples = F.interpolate(torch.stack([x[0] for x in items]).to(dev), size=scale)
            targets = [{k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                       for _, t in items]
            _train_step(solver, samples, targets, step)
        return {"batch": batch, "ok": True, **_cuda_peaks()}
    except torch.cuda.OutOfMemoryError:
        return {"batch": batch, "ok": False, "oom": True}
    finally:
        solver.optimizer.zero_grad(set_to_none=True)
        _reset_cuda()


def time_epochs(solver, n_epochs: int) -> tuple[list[float], dict]:
    """Run real D-FINE epochs (multi-scale, augmentation, EMA); return wall times and peaks."""
    import torch
    from src.solver.det_engine import train_one_epoch
    times = []
    _reset_cuda()
    for epoch in range(n_epochs):
        solver.train_dataloader.set_epoch(epoch)
        t0 = time.perf_counter()
        train_one_epoch(
            solver.model, solver.criterion, solver.train_dataloader, solver.optimizer,
            solver.device, epoch, use_wandb=False, epochs=n_epochs,
            max_norm=solver.cfg.clip_max_norm, print_freq=10**6, ema=solver.ema,
            scaler=solver.scaler, lr_warmup_scheduler=solver.lr_warmup_scheduler,
            writer=None, output_dir=None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(round(time.perf_counter() - t0, 2))
    return times, _cuda_peaks()


def environment(dfine_dir: Path, seed: int, seed_util: str) -> dict:
    import torch
    try:
        sha = subprocess.run(["git", "-C", str(dfine_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "platform": "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else
                    "colab" if "COLAB_GPU" in os.environ else "local",
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": gpu.name if gpu else None,
        "gpu_total_gb": round(gpu.total_memory / GB, 2) if gpu else None,
        "dfine_commit": sha, "seed": seed, "seed_util": seed_util,
    }


def run(args: argparse.Namespace) -> dict:
    dfine_dir = Path(args.dfine_dir).resolve()
    sys.path.insert(0, str(dfine_dir))  # D-FINE's own "src" package
    from src.misc import dist_utils

    seed_util = set_seed(args.seed)
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=args.seed)
    work = Path(args.work_dir)
    data_dir = work / "mock_coco"
    if not (data_dir / "train.json").exists():
        make_synthetic_coco(data_dir, n_images=args.n_mock, size=512, seed=args.seed)
    n_mock = len(json.loads((data_dir / "train.json").read_text())["images"])

    amp = not args.no_amp
    results = {"env": environment(dfine_dir, args.seed, seed_util),
               "settings": {"amp": amp, "peak_limit_gb": PEAK_LIMIT_GB, "num_classes": NUM_CLASSES,
                            "n_mock_images": n_mock, "n_train_for_extrapolation": args.n_train,
                            "workers": args.workers, "pretrained": not args.no_pretrained},
               "runs": []}
    for model in args.models:
        for size in args.sizes:
            scale = max_train_scale(size)
            print(f"\n=== D-FINE-{model.upper()} @ {size} (max multi-scale {scale}) ===", flush=True)
            mk = lambda b, tag: write_config(  # noqa: E731
                work / f"cfg_{model}_{size}_{tag}.yml", dfine_dir, model, size, data_dir, b,
                args.workers, amp, not args.no_pretrained, work / f"out_{model}_{size}")

            solver = build_solver(mk(args.batches[0], "probe"), args.device, args.seed, amp)
            probes = []
            for b in args.batches:
                p = probe_batch(solver, b, scale)
                probes.append(p)
                print(f"  batch {b:>3}: {p}", flush=True)
                if not p["ok"]:
                    break
            del solver
            _reset_cuda()

            ok = [p for p in probes if p["ok"]]
            within = [p for p in ok if p["peak_reserved_gb"] is None
                      or p["peak_reserved_gb"] <= PEAK_LIMIT_GB]
            largest = max((p["batch"] for p in ok), default=None)
            chosen = max((p["batch"] for p in within), default=None)
            entry = {"model": model, "size": size, "max_train_scale": scale, "probes": probes,
                     "largest_workable_batch": largest, "chosen_batch": chosen,
                     "largest_tried_batch": args.batches[-1]}
            if chosen is not None:
                set_seed(args.seed)
                solver = build_solver(mk(chosen, "epoch"), args.device, args.seed, amp)
                times, peaks = time_epochs(solver, args.epochs)
                del solver
                _reset_cuda()
                timed = times[-1]
                entry.update({
                    "epoch_times_s": times, "s_per_epoch_mock": timed,
                    "s_per_image": round(timed / n_mock, 4),
                    "s_per_epoch_extrapolated": round(timed / n_mock * args.n_train, 1)
                    if args.n_train else None,
                    "epoch_peak_alloc_gb": peaks["peak_alloc_gb"],
                    "epoch_peak_reserved_gb": peaks["peak_reserved_gb"],
                    "epoch_within_limit": peaks["peak_reserved_gb"] is None
                    or peaks["peak_reserved_gb"] <= PEAK_LIMIT_GB,
                })
            print(f"  -> {({k: v for k, v in entry.items() if k != 'probes'})}", flush=True)
            results["runs"].append(entry)
    return results


SUMMARY_FIELDS = ["model", "size", "max_train_scale", "largest_workable_batch", "chosen_batch",
                  "chosen_peak_reserved_gb", "epoch_peak_reserved_gb", "epoch_within_limit",
                  "s_per_epoch_mock", "s_per_image", "s_per_epoch_extrapolated"]


def summarise(results: dict) -> list[dict]:
    rows = []
    for r in results["runs"]:
        chosen = next((p for p in r["probes"] if p["batch"] == r["chosen_batch"]), {})
        rows.append({**{k: r.get(k) for k in SUMMARY_FIELDS},
                     "chosen_peak_reserved_gb": chosen.get("peak_reserved_gb")})
    return rows


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dfine-dir", default=str(REPO / "third_party" / "D-FINE"))
    ap.add_argument("--out", default=str(REPO / "artifacts" / "p3" / "feasibility"))
    ap.add_argument("--work-dir", default=str(Path(tempfile.gettempdir()) / "p3_dfine_feasibility"))
    ap.add_argument("--models", nargs="+", default=["s", "m"], choices=["n", "s", "m", "l", "x"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[512, 640])
    ap.add_argument("--batches", nargs="+", type=int, default=list(DEFAULT_BATCHES))
    ap.add_argument("--n-mock", type=int, default=320, help="synthetic training images")
    ap.add_argument("--n-train", type=int, default=None,
                    help="real training-set size, for s/epoch extrapolation (from C01 when known)")
    ap.add_argument("--epochs", type=int, default=2, help="epochs to run; the last one is timed")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="skip HGNetV2 weight download")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check plumbing")
    args = ap.parse_args(argv)
    if args.smoke:
        args.models, args.sizes, args.batches = ["n"], [512, 640], [2]
        args.n_mock, args.epochs, args.workers, args.device = 8, 1, 0, "cpu"
        args.no_amp, args.no_pretrained = True, True
        args.work_dir = str(Path(args.work_dir) / "smoke")
        args.out = args.work_dir  # smoke output is not an artifact
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    results = run(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "smoke_cpu" if args.smoke else "dfine_t4_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"{stem}.json").write_text(json.dumps(results, indent=2))
    rows = summarise(results)
    with open(out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out / stem}.json and .csv")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
