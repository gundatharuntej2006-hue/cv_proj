"""Read-only readiness checks. Never creates results, locks decisions, or loads sealed data.

Run: python -m p3_ops.preflight [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def inspect(root=ROOT, *, check_cuda=True, data_root=None, stage="all"):
    if stage not in ("setup", "training", "runtime", "all"):
        raise ValueError("Unknown readiness stage")
    root = Path(root)
    data_root = Path(data_root or os.environ.get("TB_DATA_ROOT", root))
    checks = []

    def add(name, ready, detail):
        checks.append({"check": name, "ready": bool(ready), "detail": detail})

    from p3_det.setup import PIN, setup
    try:
        setup(root / "third_party/D-FINE")
        registered = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--stage", "third_party/D-FINE"], text=True).strip()
        add("D-FINE pinned submodule", registered.startswith(f"160000 {PIN}"),
            "Pinned checkout and gitlink required; commit .gitmodules and the submodule registration")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        add("D-FINE pinned submodule", False, str(exc))
    add("real mode", os.environ.get("TB_MOCK") != "1" and os.environ.get("TB_DET_ALLOW_UNLOCKED") != "1",
        "Reported work must not use mocks or unlocked detector mode")
    if stage == "setup":
        return checks
    for name in ("data/splits/tbx11k_splits_v1.csv", "data/processed/v1/manifest.parquet",
                 "data/processed/v1/lung_roi.parquet"):
        add(name, (data_root / name).is_file(), f"Required P1 input under {data_root}")
    add("original development lesion boxes", (data_root / "data/boxes/v1/coco_development_full.json").is_file(),
        "All original boxes required: cropped annotations omit out-of-crop lesions")
    names = ["p2_clahe", "p2_crop_margin"]
    if stage in ("runtime", "all"):
        names += ["p3_det_input_size", "p3_det_checkpoint", "p3_det_conf_threshold"]
    if stage == "all":
        names += ["p3_gradcam_config", "p3_licences"]
    for name in names:
        path = root / "decisions" / f"{name}.json"
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            add(name, d.get("locked") and bool(d.get("value")),
                f"{d.get('decision_id')}: measured/approved value and lock required")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            add(name, False, type(exc).__name__)
    from p3_det.decisions import verify_order
    for name in ("p3_det_input_size", "p3_det_checkpoint", "p3_det_conf_threshold", "p3_gradcam_config"):
        if stage == "runtime" and name == "p3_gradcam_config":
            continue
        ready, detail = verify_order(root / "decisions" / f"{name}.json")
        add(f"{name} history", ready, detail.splitlines()[-1])
    import yaml
    for size in ((512, 640) if stage in ("training", "all") else ()):
        path = root / f"src/p3_det/configs/dfine_{size}.yaml"
        try:
            schedule = yaml.safe_load(path.read_text())["schedule"]
            add(f"training schedule {size}", all(schedule.get(k) for k in ("epochs", "batch_size")),
                "Fill from measured T4 feasibility; do not guess these values")
        except (OSError, ValueError, KeyError) as exc:
            add(f"training schedule {size}", False, str(exc))
    try:
        from p3_det.data.config import resolve_preprocessing
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            margin, _, provisional = resolve_preprocessing(decisions_dir=root / "decisions")
        if not provisional:
            for split in ("train", "development"):
                name = f"data/boxes/v1/coco_{split}_m{margin:.2f}.json"
                add(name, (data_root / name).is_file(), "C06 at locked D09 margin")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        add("C06 paths", False, str(exc))
    if check_cuda:
        try:
            import torch
            add("CUDA", torch.cuda.is_available(), f"torch {torch.__version__}")
        except ImportError:
            add("CUDA", False, "PyTorch is not installed")
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--stage", choices=("setup", "training", "runtime", "all"), default="all")
    ap.add_argument("--data-root", type=Path, help="P1 contract root; also supported through TB_DATA_ROOT")
    ap.add_argument("--no-cuda", action="store_true", help="Inspect CPU readiness without claiming GPU acceptance")
    a = ap.parse_args(argv)
    checks = inspect(data_root=a.data_root, stage=a.stage, check_cuda=not a.no_cuda)
    if a.json:
        print(json.dumps(checks, indent=2))
    else:
        for c in checks:
            print(f"{'READY' if c['ready'] else 'MISSING'}: {c['check']} - {c['detail']}")
    return 0 if all(c["ready"] for c in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
