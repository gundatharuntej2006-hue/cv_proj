"""P3-10: C11 for DEVELOPMENT: Grad-CAM heatmaps + localisation table, for two classifiers.

    python -m p3_xai.export_c11 --crop-model <id> --full-model <id> [--limit N]

* crop model (primary, D11)  : heatmap on the lung crop (C05 at the D09 margin), as at runtime.
* full model (comparator)    : heatmap on the whole 512 image (crop = [0, 0, 512, 512]).
Both model ids must appear in P2's model registry (D14, decisions/p2_model_registry.json) when it exists.
Images: development TB images with annotated boxes (C01 + all original boxes), readable (C04 per D08).

Writes, per model:  artifacts/p3/xai/{model_id}/development/{image_id}.npy
                    artifacts/p3/xai/loc_{model_id}_development_v1.parquet
and the summary     artifacts/p3/xai/loc_summary_development.json (n + bootstrap 95% CIs).
Heatmaps are ~0.5 MB each: keep them on Drive, not in git (README: no large files in the repo).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from p3_det import decisions as dec
from p3_det.data.boxes import original_boxes
from p3_det.data.config import DataPaths, check_split, resolve_preprocessing
from p3_det.metrics import MetricResult, bootstrap_ci, bootstrap_settings

from . import c11
from .gradcam import GradCamExplainer
from .metrics import SUBTYPE_FROM_C01, localisation_row

SPLIT = "development"
OUT_ROOT = dec.REPO / "artifacts" / "p3" / "xai"
REGISTRY = dec.REPO / "decisions" / "p2_model_registry.json"
FULL_CROP = [0, 0, 512, 512]


@dataclass(frozen=True)
class ModelRun:
    model_id: str
    role: str                                  # "crop" (primary) or "full" (full-image comparator)
    provider: Callable


def p2_provider(model_id: str, role: str) -> Callable:
    """Load a classifier from P2's R2. classifier_module() serves the primary (D11) model; R2 has no
    defined way to ask for another model (docs/p3_contract_notes.md item 14), so the full-image model
    is requested with classifier_module(model_id=...) and fails clearly if R2 does not support it."""
    def load():
        try:
            from p2_cls.api import classifier_module
        except ImportError as e:
            raise RuntimeError("p2_cls.api.classifier_module (R2) is not available") from e
        if role == "crop":
            m = classifier_module()
        else:
            try:
                m = classifier_module(model_id=model_id)
            except TypeError as e:
                raise RuntimeError(f"R2 cannot load '{model_id}': classifier_module() takes no model_id "
                                   "(contract notes item 14)") from e
        got = getattr(m, "model_id", None)
        if got is not None and got != model_id:
            raise RuntimeError(f"R2 returned model '{got}', expected '{model_id}'")
        return m
    return load


def _registry_strings(obj) -> set[str]:
    """Every string (keys and values) in the D14 value; D14's exact layout is P2's to define."""
    if isinstance(obj, str):
        return {obj}
    children = [*obj.keys(), *obj.values()] if isinstance(obj, dict) else obj if isinstance(obj, list) else []
    out: set[str] = set()
    for c in children:
        out |= _registry_strings(c)
    return out


def check_registry(model_ids: list[str], registry: Path = REGISTRY) -> str:
    if not registry.exists():
        warnings.warn("D14 model registry not available: model ids are not verified against it", stacklevel=2)
        return "unverified (no D14)"
    known = _registry_strings(json.loads(registry.read_text()).get("value", {}))
    missing = [m for m in model_ids if m not in known]
    if missing:
        raise dec.Refused(f"model ids not in P2's registry (D14): {missing}")
    return "D14"


def development_items(paths: DataPaths, decisions_dir: Path | None = None, dataset: str = "tbx11k") -> list[dict]:
    """Development TB images with >= 1 original box: image path, lung crop, boxes, subtype."""
    import pyarrow.parquet as pq
    check_split(SPLIT, dataset)
    margin, clahe, _ = resolve_preprocessing(decisions_dir=decisions_dir)
    with open(paths.splits, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == SPLIT and int(r["tb_label"]) == 1]
    man = {r["image_id"]: r for r in pq.read_table(paths.manifest).to_pylist() if r["dataset"] == dataset}
    roi = {r["image_id"]: r for r in pq.read_table(paths.lung_roi).to_pylist() if abs(r["margin"] - margin) < 1e-6}
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        gt = original_boxes(paths, SPLIT)
    items = []
    for r in rows:
        iid = r["image_id"]
        m, g = man.get(iid), gt.get(iid)
        if m is None or m["decode_status"] != "ok" or g is None or not len(g["boxes"]):
            continue
        subtype = SUBTYPE_FROM_C01.get(r["category"])
        if subtype is None:
            s = {x for x in g["subtypes"] if x}
            subtype = "active+latent" if s >= {"active", "latent"} else (next(iter(s)) if len(s) == 1 else None)
        if subtype is None:
            raise ValueError(f"{iid}: cannot determine tb_subtype (C01 category {r['category']!r})")
        rel = Path(m["path_img512_clahe"] if clahe else m["path_img512"])
        ro = roi[iid]
        items.append({"image_id": iid, "path": rel if rel.is_absolute() else paths.root / rel,
                      "crop_box": [ro["crop_x0"], ro["crop_y0"], ro["crop_x1"], ro["crop_y1"]],
                      "boxes": g["boxes"], "tb_subtype": subtype})
    return items


def export_model(run: ModelRun, items: list[dict], out_root: Path = OUT_ROOT) -> Path:
    explainer = GradCamExplainer(run.provider())
    rows = []
    for it in items:
        img = np.asarray(Image.open(it["path"]).convert("L"))
        crop = it["crop_box"] if run.role == "crop" else FULL_CROP
        h = explainer(img, crop)
        hp = c11.heatmap_path(out_root, run.model_id, SPLIT, it["image_id"])
        hp.parent.mkdir(parents=True, exist_ok=True)
        np.save(hp, h)
        c11.validate_heatmap(hp, crop)
        rows.append(localisation_row(it["image_id"], run.model_id, h, it["boxes"], it["tb_subtype"]).__dict__)
    return c11.write_table(c11.table_path(out_root, run.model_id, SPLIT), rows)


def _result(name: str, values: np.ndarray, n_boxes: int, B: int, seed: int) -> MetricResult:
    lo, hi, _ = bootstrap_ci(len(values), lambda idx: float(values[idx].mean()), B=B, seed=seed)
    return MetricResult(name, float(values.mean()), lo, hi, n_images=len(values), n_tb_images=len(values),
                        n_non_tb_images=0, n_boxes=n_boxes, ci_method="bootstrap_percentile")


def summarise(tables: dict[str, Path], *, crop_model: str | None = None, full_model: str | None = None) -> dict:
    """Pointing accuracy and mean energy-in-box per model (n + bootstrap 95% CI), and the paired
    crop - full difference on the images both models share."""
    import pyarrow.parquet as pq
    B, seed, source = bootstrap_settings()
    data = {m: pq.read_table(p).to_pydict() for m, p in tables.items()}
    out = {"split": SPLIT, "bootstrap": {"B": B, "seed": seed, "source": source}, "models": {}}
    for m, d in data.items():
        hit = np.asarray(d["pointing_hit"], float)
        en = np.asarray(d["energy_in_box"], float)
        nb = int(sum(d["n_gt_boxes"]))
        out["models"][m] = {"pointing_accuracy": _result("pointing_accuracy", hit, nb, B, seed).as_dict(),
                            "mean_energy_in_box": _result("mean_energy_in_box", en, nb, B, seed).as_dict()}
    if crop_model and full_model:
        a, b = data[crop_model], data[full_model]
        ia = {i: k for k, i in enumerate(a["image_id"])}
        common = [(ia[i], k) for k, i in enumerate(b["image_id"]) if i in ia]
        if common:
            ka, kb = map(np.asarray, zip(*common))
            nb = int(sum(a["n_gt_boxes"][k] for k in ka))
            for metric, col in (("pointing_accuracy", "pointing_hit"), ("mean_energy_in_box", "energy_in_box")):
                diff = np.asarray(a[col], float)[ka] - np.asarray(b[col], float)[kb]
                out[f"{metric}_crop_minus_full"] = _result(f"{metric}_crop_minus_full", diff, nb, B, seed).as_dict()
    return out


def print_summary(s: dict) -> None:
    def fmt(r):
        return f"{r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
    print(f"\nLocalisation agreement, {s['split']} (bootstrap 95% CI over images, B={s['bootstrap']['B']})")
    print(f"{'model':40s} {'n img':>6s} {'n box':>6s}  {'pointing accuracy':24s} {'mean energy-in-box':24s}")
    for m, r in s["models"].items():
        p, e = r["pointing_accuracy"], r["mean_energy_in_box"]
        print(f"{m:40s} {p['n_images']:6d} {p['n_boxes']:6d}  {fmt(p):24s} {fmt(e):24s}")
    for k in ("pointing_accuracy_crop_minus_full", "mean_energy_in_box_crop_minus_full"):
        if k in s:
            print(f"{k:40s} {s[k]['n_images']:6d} {s[k]['n_boxes']:6d}  {fmt(s[k])} (paired)")
    print("Agreement with incomplete radiologist boxes; attention evidence, not a finding.")


def export(runs: list[ModelRun], *, paths: DataPaths | None = None, out_root: Path = OUT_ROOT,
           decisions_dir: Path | None = None, registry: Path = REGISTRY, limit: int | None = None) -> dict:
    if {r.role for r in runs} - {"crop", "full"}:
        raise ValueError("role must be 'crop' or 'full'")
    reg = check_registry([r.model_id for r in runs], registry)
    items = development_items(paths or DataPaths.default(), decisions_dir)
    items = items[:limit] if limit else items
    tables = {r.model_id: export_model(r, items, out_root) for r in runs}
    crop = next((r.model_id for r in runs if r.role == "crop"), None)
    full = next((r.model_id for r in runs if r.role == "full"), None)
    s = summarise(tables, crop_model=crop, full_model=full)
    s["registry_check"] = reg
    s["tables"] = {m: dec.repo_rel(p) for m, p in tables.items()}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "loc_summary_development.json").write_text(json.dumps(s, indent=1))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crop-model", required=True, help="primary lung-crop model id (D11, in D14)")
    ap.add_argument("--full-model", required=True, help="full-image comparator model id (in D14)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    try:
        s = export([ModelRun(a.crop_model, "crop", p2_provider(a.crop_model, "crop")),
                    ModelRun(a.full_model, "full", p2_provider(a.full_model, "full"))], limit=a.limit)
    except (dec.Refused, RuntimeError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print_summary(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
