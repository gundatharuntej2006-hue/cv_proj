"""P3-08: export C10 detections for the development and calibration splits with R3 detect().

    python -m p3_det.export_c10                      # both splits -> artifacts/p3/det/detections_tbx11k_{split}_v1.json
    python -m p3_det.export_c10 --split development

Every readable image of the split (C01), TB or not, gets an entry: P4 needs all of them for the
link rule and the calibration operating curve. Images are the C04 variant chosen by D08 and crops
come from C05 at the D09 margin, i.e. exactly what R1 hands to detect() at runtime. Sealed splits
(internal_test, challenge_test, external_test) are refused here; P3-13 exports them through the gate.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from . import api, c10
from .data.config import DataPaths, check_split, resolve_preprocessing

REPO = Path(__file__).resolve().parents[2]
EXPORT_SPLITS = ("development", "calibration")
OUT_DIR = REPO / "artifacts" / "p3" / "det"


def split_images(paths: DataPaths, split: str, dataset: str, margin: float, clahe: bool):
    """(image_id, image path, crop_box) for every readable image of the split, in C01 order."""
    import pyarrow.parquet as pq
    with open(paths.splits, newline="") as f:
        ids = [r["image_id"] for r in csv.DictReader(f) if r["split"] == split]
    manifest = {r["image_id"]: r for r in pq.read_table(paths.manifest).to_pylist() if r["dataset"] == dataset}
    roi = {r["image_id"]: r for r in pq.read_table(paths.lung_roi).to_pylist() if abs(r["margin"] - margin) < 1e-6}
    out, skipped = [], 0
    for iid in ids:
        m = manifest.get(iid)
        if m is None or m["decode_status"] != "ok":
            skipped += 1
            continue
        ro = roi[iid]
        rel = Path(m["path_img512_clahe"] if clahe else m["path_img512"])
        out.append((iid, rel if rel.is_absolute() else paths.root / rel,
                    [ro["crop_x0"], ro["crop_y0"], ro["crop_x1"], ro["crop_y1"]]))
    return out, skipped


def export(split: str, *, dataset: str = "tbx11k", paths: DataPaths | None = None, out_dir: Path = OUT_DIR,
           decisions_dir: Path | None = None) -> Path:
    if split not in EXPORT_SPLITS:
        raise PermissionError(f"export_c10 writes only {EXPORT_SPLITS}; '{split}' is not allowed here")
    check_split(split, dataset)
    paths = paths or DataPaths.default()
    margin, clahe, provisional = resolve_preprocessing(decisions_dir=decisions_dir)
    if provisional:
        warnings.warn("D08/D09 not locked: C10 exported on PROVISIONAL preprocessing", stacklevel=2)
    info = api.detector_info()
    if not info["mock"]:
        trained = api._detector().s.run_meta["preprocessing"]
        if (trained["margin"], trained["clahe"]) != (margin, clahe):
            raise RuntimeError(f"detector was trained with margin={trained['margin']}, clahe={trained['clahe']} "
                               f"but D08/D09 now give margin={margin}, clahe={clahe}")
    items, skipped = split_images(paths, split, dataset, margin, clahe)
    entries, t0 = [], time.time()
    for iid, img_path, crop in items:
        img = np.asarray(Image.open(img_path).convert("L"))
        r = api.detect(img, crop)
        entries.append(c10.entry(iid, crop, r.boxes_full512, r.scores, info["conf_threshold"]))
    doc = c10.document(info["model_id"], info["input_size"], info["conf_threshold"], entries)
    path = c10.write(Path(out_dir) / f"detections_{dataset}_{split}_v1.json", doc)
    print(f"[export_c10] {split}: {len(entries)} images ({skipped} unreadable skipped) in {time.time() - t0:.0f}s "
          f"-> {path}  model={info['model_id']} thr={info['conf_threshold']}")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=EXPORT_SPLITS, action="append")
    ap.add_argument("--dataset", default="tbx11k")
    a = ap.parse_args(argv)
    for split in a.split or EXPORT_SPLITS:
        export(split, dataset=a.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
