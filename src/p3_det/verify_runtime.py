"""P3-08: compare a real locked detector with C10 on 50 development images.

python -m p3_det.verify_runtime --device cuda
Records actual runtime/export agreement and latency; refuses mocks, provisional
preprocessing, mismatched model provenance and incomplete samples. No sealed data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import api, c10
from .data.config import DataPaths, resolve_preprocessing
from .export_c10 import split_images

ROOT = Path(__file__).resolve().parents[2]


def compare_entry(expected, actual, *, coord_atol=0.0011, score_atol=1e-5):
    """Compare unordered boxes; tolerate C10's documented 0.001-pixel rounding."""
    if expected["crop_box"] != actual["crop_box"]:
        return False
    if expected["n_lesions_above_thr"] != actual["n_lesions_above_thr"]:
        return False
    a = np.asarray(sorted(expected["boxes"]), float).reshape(-1, 5)
    b = np.asarray(sorted(actual["boxes"]), float).reshape(-1, 5)
    return bool(a.shape == b.shape and np.allclose(a[:, :4], b[:, :4], atol=coord_atol, rtol=0)
                and np.allclose(a[:, 4], b[:, 4], atol=score_atol, rtol=0))


def verify(*, device="cuda", source=None, out=None, paths=None):
    if os.environ.get("TB_MOCK") == "1" or os.environ.get("TB_DET_ALLOW_UNLOCKED") == "1":
        raise PermissionError("Runtime acceptance refuses mock/unlocked modes")
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda")
    os.environ["TB_DET_DEVICE"] = device
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU substitution for the GPU acceptance test")
    margin, clahe, provisional = resolve_preprocessing()
    if provisional:
        raise PermissionError("Runtime acceptance requires locked D08/D09")
    info = api.detector_info()
    if info.get("mock") or info.get("locked") is not True:
        raise PermissionError("Runtime acceptance requires a real locked detector")
    source = Path(source or ROOT / "artifacts/p3/det/detections_tbx11k_development_v1.json")
    c10.validate_c10(source)
    doc = json.loads(source.read_text())
    for key in ("model_id", "input_size", "conf_threshold"):
        if doc[key] != info[key]:
            raise ValueError(f"C10 {key} differs from the loaded detector")
    expected = {row["image_id"]: row for row in doc["images"]}
    items, _ = split_images(paths or DataPaths.default(), "development", "tbx11k", margin, clahe)
    if len(items) < 50:
        raise ValueError("At least 50 readable development images are required")
    items = [items[int(i)] for i in np.random.default_rng(0).choice(len(items), size=50, replace=False)]
    if any(iid not in expected for iid, _, _ in items):
        raise ValueError("C10 does not cover the selected development sample")
    timings, rows = [], []
    for i, (iid, path, crop) in enumerate(items):
        image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        if i == 0:
            for _ in range(3):
                api.detect(image, crop)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = api.detect(image, crop)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        entry = c10.entry(iid, crop, result.boxes_full512, result.scores, info["conf_threshold"])
        rows.append({"image_id": iid, "matches_c10": compare_entry(expected[iid], entry)})
    median, p95 = map(float, (np.median(timings), np.percentile(timings, 95)))
    report = {"split": "development", "n": 50, "seed": 0, "warmup": 3, "device": device,
              "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
              "detector": info, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "all_match": all(row["matches_c10"] for row in rows), "per_image": rows,
              "median_ms": median, "p95_ms": p95,
              "gpu_latency_pass": median <= 500 if device == "cuda" else None,
              "scope": "Detector only; not full-pipeline E6"}
    out = Path(out or ROOT / f"artifacts/p3/verify/detect_runtime_{device}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = verify(device=args.device, source=args.source, out=args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))
    return 0 if report["all_match"] and report["gpu_latency_pass"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
