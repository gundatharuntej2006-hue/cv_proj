"""One-shot P3 C10/C11 export under an existing P4 gate token.

Never run this during development. All D01-D23 must already be locked by their
owners. The internal split has been exposed in an earlier P2 rehearsal: this
runner does not make it an untouched test set. No token is issued by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from p3_det import api as detector, c10
from p3_det.data.boxes import original_boxes
from p3_det.data.config import DataPaths, resolve_preprocessing
from p3_det.export_c10 import split_images
from p3_xai import api as xai, c11
from p3_xai.metrics import localisation_row, SUBTYPE_FROM_C01
from . import _tbcore_adapter as shared

ROOT = Path(__file__).resolve().parents[2]


def run(split, *, dataset="tbx11k", model_id, full_model_id=None, paths=None, out_root=None, token_dir=None):
    model_ids = [model_id] + ([full_model_id] if full_model_id else [])
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", mid) for mid in model_ids):
        raise ValueError("Unsafe model id")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("Primary and full-image comparator must have different model ids")
    if dataset != "tbx11k":
        raise ValueError("This runner supports the TBX11K C01 contract only")
    if os.environ.get("TB_MOCK") == "1" or os.environ.get("TB_DET_ALLOW_UNLOCKED") == "1":
        raise PermissionError("Sealed export refuses mock and unlocked-model modes")
    if split not in ("internal_test", "challenge_test", "external_test"):
        raise ValueError("A sealed split is required")
    code_paths = ["src", "decisions", "contracts"]
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", *code_paths]).returncode
    untracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard",
                                        "--", *code_paths], text=True).strip()
    if dirty or untracked:
        raise PermissionError("Commit reviewed code and decisions before a sealed run; token SHA must identify the code")
    paths = paths or DataPaths.default()
    out_root = Path(out_root or ROOT / "artifacts/p3")
    gate = Path(token_dir or ROOT / "artifacts/p4/gate")
    if split == "challenge_test":
        prior = json.loads((out_root / "sealed/internal_test.json").read_text())
        if prior.get("status") != "complete":
            raise PermissionError("Complete internal_test before challenge_test")
    shared.validate_gate()
    from p3_xai.export_c11 import check_registry, p2_provider
    if check_registry(model_ids).startswith("unverified"):
        raise PermissionError("A real D14 model registry is required")
    margin, clahe, provisional = resolve_preprocessing()
    if provisional:
        raise PermissionError("Sealed export requires locked preprocessing")
    info = detector.detector_info()  # verifies checkpoint before consuming a token
    if info.get("mock") or not info.get("locked"):
        raise PermissionError("Sealed export requires the real locked detector")
    explainer = xai.explainer()  # requires a real P2 provider before the token is consumed
    explainers = {model_id: explainer}
    if full_model_id:
        from p3_xai.gradcam import GradCamExplainer
        explainers[full_model_id] = GradCamExplainer(p2_provider(full_model_id, "full")())
    explanation_specs = {}
    for mid, loaded in explainers.items():
        loaded_id = getattr(getattr(loaded, "module", None), "model_id", None)
        if loaded_id != mid:
            raise PermissionError(f"Loaded explanation model {loaded_id!r} does not match registered model {mid!r}")
        explanation_specs[mid] = loaded.spec.as_record()
    # Provider/specification errors must occur before the single-use token is consumed.
    json.dumps(explanation_specs, allow_nan=False)
    if split != "challenge_test" and not (paths.root / f"data/boxes/v1/coco_{split}_full.json").is_file():
        raise FileNotFoundError("All original full-image boxes are required for sealed localisation")
    sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    audit_path = out_root / "sealed" / f"{split}.json"
    if audit_path.exists():
        raise FileExistsError("A P3 sealed run record already exists; no repeat evaluation")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with shared.sealed_scope(dataset, split, gate, sha) as token:
        audit = {"status": "started", "split": split, "dataset": dataset, "git_sha": sha,
                 "token_id": token["token_id"], "detector": info, "xai_model_id": model_id,
                 "explanation_spec": explanation_specs[model_id], "explanation_specs": explanation_specs}
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        try:
            images, skipped = split_images(paths, split, dataset, margin, clahe)
            if not images:
                raise ValueError("No readable images in the selected sealed split")
            # Challenge labels/boxes are withheld. Export predictions and heatmaps
            # without trying to manufacture or read challenge ground truth.
            gt = {} if split == "challenge_test" else original_boxes(paths, split)
            with paths.splits.open(newline="") as f:
                labels = {r["image_id"]: r for r in csv.DictReader(f) if r["split"] == split}
            entries, rows = [], {mid: [] for mid in model_ids}
            heatmap_paths = []
            for iid, path, crop in images:
                img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
                result = detector.detect(img, crop)
                entries.append(c10.entry(iid, crop, result.boxes_full512, result.scores, info["conf_threshold"]))
                boxes = gt.get(iid, {}).get("boxes", [])
                subtype = None
                if len(boxes):
                    subtype = SUBTYPE_FROM_C01.get(labels[iid].get("category"))
                    if subtype is None:
                        from p3_det.train.run import image_subtype
                        subtype = image_subtype(gt[iid].get("subtypes", []))
                    if subtype is None:
                        raise ValueError(f"{iid}: unknown localisation subtype")
                for mid, loaded in explainers.items():
                    model_crop = [0, 0, 512, 512] if mid == full_model_id else crop
                    heatmap = xai.explain(img, crop) if mid == model_id else loaded(img, model_crop)
                    hp = c11.heatmap_path(out_root / "xai", mid, split, iid)
                    hp.parent.mkdir(parents=True, exist_ok=True)
                    np.save(hp, heatmap)
                    c11.validate_heatmap(hp, model_crop)
                    heatmap_paths.append(hp)
                    if len(boxes):
                        rows[mid].append(asdict(localisation_row(iid, mid, heatmap, boxes, subtype)))
            detections = c10.write(out_root / "det" / f"detections_{dataset}_{split}_v1.json",
                c10.document(info["model_id"], info["input_size"], info["conf_threshold"], entries))
            tables = {mid: c11.write_table(c11.table_path(out_root / "xai", mid, split), rows[mid])
                      for mid in model_ids}
            audit.update(status="complete", n_images=len(images), n_skipped=skipped,
                         detections=str(detections), localisation=str(tables[model_id]),
                         localisation_tables={mid: str(path) for mid, path in tables.items()},
                         localisation_evaluable=split != "challenge_test")
            produced = [detections, *tables.values(), *heatmap_paths]
            audit["artifact_sha256"] = {
                str(p.relative_to(out_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in produced}
        except BaseException as exc:
            audit.update(status="failed", error=type(exc).__name__)
            raise
        finally:
            audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("internal_test", "challenge_test", "external_test"), required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--full-model-id", help="Registered full-image comparator, exported under the same token")
    ap.add_argument("--dataset", default="tbx11k")
    a = ap.parse_args()
    print(run(a.split, dataset=a.dataset, model_id=a.model_id, full_model_id=a.full_model_id))


if __name__ == "__main__":
    main()
