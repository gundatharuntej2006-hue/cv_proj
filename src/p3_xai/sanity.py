"""P3-09: D19 model-randomisation sanity check (Adebayo et al. 2018) on development images.

    python -m p3_xai.sanity                        # P2's classifier_module(), 20 development images
    python -m p3_xai.sanity --write-d19 [--lock]   # also fill decisions/p3_gradcam_config.json

For each of 20 development images (seeded: 10 TB + 10 non-TB, readable, crop at the D09 margin, C04
variant per D08) the Grad-CAM heatmap of the trained classifier is compared with the heatmap of a
copy whose head (and, reported too, head + last block) is re-initialised. Spearman correlation is
computed over the pixels INSIDE the crop (outside both maps are zero by construction, which would
inflate it). Pass (D19, committed 7a812a7): mean |Spearman| (head re-initialised) < 0.5; the maximum
|Spearman| and every per-image value are reported too. The absolute value is used because a signed rule
would pass an inverted heatmap, which still depends on the weights. A constant randomised map gives an
undefined correlation; it is counted as 0 (no dependence left) and reported separately.

Results: artifacts/p3/xai/sanity/randomisation_<model_id>.json (+ D19 value with --write-d19).
A run on P2's stub or a stand-in is saved under artifacts/p3/xai/sanity/stub_not_evidence/ with
"evidence_status": "stub, not evidence"; it can never be written into D19.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from p3_det import decisions as dec
from p3_det.data.config import DataPaths, check_split, resolve_preprocessing

from .d19 import CRITERION, D19_PATH
from .gradcam import GradCamExplainer, randomised_copy

SPLIT = "development"
OUT_DIR = dec.REPO / "artifacts" / "p3" / "xai" / "sanity"
VRAM_FILE = dec.REPO / "artifacts" / "p3" / "xai" / "vram_explain.json"
VRAM_LIMIT_GB = 5.5
PASS_BELOW = 0.5
STUB_DIR = OUT_DIR / "stub_not_evidence"
STUB_LABEL = "stub, not evidence"


def sample_images(paths: DataPaths, n_tb: int = 10, n_non_tb: int = 10, seed: int = 0,
                  decisions_dir: Path | None = None, dataset: str = "tbx11k"):
    """Seeded sample of development images: [(image_id, tb_label, image path, crop_box)]."""
    import pyarrow.parquet as pq
    check_split(SPLIT, dataset)
    margin, clahe, _ = resolve_preprocessing(decisions_dir=decisions_dir)
    with open(paths.splits, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == SPLIT]
    man = {r["image_id"]: r for r in pq.read_table(paths.manifest).to_pylist() if r["dataset"] == dataset}
    roi = {r["image_id"]: r for r in pq.read_table(paths.lung_roi).to_pylist() if abs(r["margin"] - margin) < 1e-6}
    ok = [r for r in rows if r["image_id"] in man and man[r["image_id"]]["decode_status"] == "ok"]
    rng = np.random.default_rng(seed)
    out = []
    for label, n in ((1, n_tb), (0, n_non_tb)):
        pool = sorted(r["image_id"] for r in ok if int(r["tb_label"]) == label)
        if len(pool) < n:
            raise RuntimeError(f"only {len(pool)} development images with tb_label={label}, need {n}")
        for iid in rng.choice(pool, size=n, replace=False):
            m, ro = man[iid], roi[iid]
            rel = Path(m["path_img512_clahe"] if clahe else m["path_img512"])
            out.append((str(iid), label, rel if rel.is_absolute() else paths.root / rel,
                        [ro["crop_x0"], ro["crop_y0"], ro["crop_x1"], ro["crop_y1"]]))
    return out


def spearman_inside(a: np.ndarray, b: np.ndarray, crop_box) -> float:
    from scipy.stats import spearmanr
    x0, y0, x1, y1 = crop_box
    u = a[y0:y1, x0:x1].astype(np.float32).ravel()
    v = b[y0:y1, x0:x1].astype(np.float32).ravel()
    if u.std() == 0 or v.std() == 0:
        return math.nan
    return float(spearmanr(u, v).statistic)


def run(module, items, scopes=("head", "head+last_block"), seed: int = 0) -> dict:
    base = GradCamExplainer(module)
    rand = {s: GradCamExplainer(randomised_copy(module, s, seed)) for s in scopes}
    per_image = []
    for iid, label, path, crop in items:
        img = np.asarray(Image.open(path).convert("L"))
        h0 = base(img, crop)
        row = {"image_id": iid, "tb_label": label, "crop_box": crop}
        for s, e in rand.items():
            v = spearman_inside(h0, e(img, crop), crop)
            row[f"spearman_{s}"], row[f"abs_spearman_{s}"] = v, (0.0 if math.isnan(v) else abs(v))
        per_image.append(row)
    return {"spec": base.spec.as_record(), "seed": seed, "pass_below": PASS_BELOW,
            "rule": "mean |Spearman| (head re-initialised) < 0.5", **evaluate(per_image, scopes)}


def evaluate(per_image: list[dict], scopes=("head", "head+last_block")) -> dict:
    """D19 rule on per-image signed values: pass if mean |rho| (head) < PASS_BELOW; undefined counts as 0."""
    summary = {}
    for s in scopes:
        vals = [r[f"spearman_{s}"] for r in per_image]
        n_nan = sum(math.isnan(v) for v in vals)
        absval = np.abs([0.0 if math.isnan(v) else v for v in vals])
        summary[s] = {"mean_abs": round(float(np.mean(absval)), 4), "max_abs": round(float(np.max(absval)), 4),
                      "median_abs": round(float(np.median(absval)), 4),
                      "per_image_abs": [round(float(v), 4) for v in absval],
                      # signed mean reported for transparency only; it is NOT the rule (an inverted map passes it)
                      "mean_signed": round(float(np.mean([0.0 if math.isnan(v) else v for v in vals])), 4),
                      "n_images": len(vals), "n_undefined_as_0": n_nan}
    return {"passed": summary["head"]["mean_abs"] < PASS_BELOW, "summary": summary, "per_image": per_image}


def _model_id(module) -> str:
    return str(getattr(module, "model_id", None) or type(module).__name__)


def _is_stub(module) -> bool:
    return bool(getattr(module, "is_stub", False)) or "stub" in _model_id(module).lower() \
        or "standin" in _model_id(module).lower()


def write_d19(results: dict, module, results_path: Path, *, lock: bool = False, dry_run: bool = False,
              path: Path = D19_PATH, vram_file: Path = VRAM_FILE, today: str | None = None) -> dict:
    if _is_stub(module) and not dry_run:
        raise dec.Refused(f"classifier is P2's stub / a stand-in: its sanity result is '{STUB_LABEL}' and is never "
                          "written into D19")
    doc = dec.load_for_result(path, CRITERION)
    sha = dec.criterion_commit(path, CRITERION)
    if sha is None and not dry_run:
        raise dec.Refused("the D19 criterion is not committed yet: commit decisions/p3_gradcam_config.json first")
    vram = json.loads(vram_file.read_text()) if vram_file.exists() else None
    doc["value"] = {
        "classifier_model_id": _model_id(module), "classifier_is_stub": _is_stub(module),
        **({"evidence_status": STUB_LABEL} if _is_stub(module) else {}),
        **results["spec"],
        "reshape_rule": f"drop first {results['spec']['n_prefix_tokens']} tokens (CLS + registers), "
                        f"reshape to {results['spec']['grid_hw'][0]}x{results['spec']['grid_hw'][1]}",
        "target": f"TB logit (output {results['spec']['tb_index']} of {results['spec']['n_outputs']})",
        "mapping": "CAM -> crop size (bilinear) -> min-max [0,1] in crop -> zero 512x512 canvas at crop_box, float16",
        "method": "pytorch-grad-cam 1.5.7 GradCAM",
        "sanity_randomisation": {"passed": results["passed"], **results["summary"], "seed": results["seed"]},
        "vram_peak_gb": (vram or {}).get("peak_gb"), "vram_device": (vram or {}).get("device"),
        "criterion_commit": sha,
    }
    doc["evidence"] = [dec.repo_rel(results_path)] + ([dec.repo_rel(vram_file)] if vram else [])
    if lock:
        problems = []
        if dry_run or sha is None:
            problems.append("criterion commit not found (or dry run)")
        if _is_stub(module):
            problems.append("classifier is P2's stub / a stand-in; D19 locks only on the real classifier (D11)")
        if not results["passed"]:
            problems.append("randomisation sanity check failed")
        if vram is None or vram["peak_gb"] > VRAM_LIMIT_GB:
            problems.append(f"VRAM not measured on the RTX 4050 or above {VRAM_LIMIT_GB} GB")
        if problems:
            raise dec.Refused("cannot lock D19: " + "; ".join(problems))
        doc.update(locked=True, locked_at=today or date.today().isoformat(), git_sha=dec.head_sha(path))
    dec.save(path, doc)
    return doc


def save_results(results: dict, module, out_dir: Path = OUT_DIR) -> Path:
    """Real classifier -> out_dir/randomisation_<id>.json; stub/stand-in -> out_dir/stub_not_evidence/ (labelled)."""
    stub = _is_stub(module)
    folder = out_dir / "stub_not_evidence" if stub else out_dir
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / (f"STUB_randomisation_{_model_id(module)}.json" if stub else f"randomisation_{_model_id(module)}.json")
    out.write_text(json.dumps({"model_id": _model_id(module), "classifier_is_stub": stub,
                               "evidence_status": STUB_LABEL if stub else "development result", **results}, indent=1))
    return out


def summary_text(results: dict) -> str:
    h, b = results["summary"]["head"], results["summary"]["head+last_block"]
    per = ", ".join(f"{v:.2f}" for v in h["per_image_abs"])
    return (f"|Spearman| inside crop, n={h['n_images']}: head re-init mean {h['mean_abs']:.3f}, max {h['max_abs']:.3f} "
            f"(signed mean {h['mean_signed']:.3f}, not the rule); head+last block mean |rho| {b['mean_abs']:.3f} -> "
            f"{'PASS' if results['passed'] else 'FAIL'} (mean |rho| < {PASS_BELOW}); per image: [{per}]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-d19", action="store_true")
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from . import api
    module = api.explainer().module
    items = sample_images(DataPaths.default(), seed=a.seed)
    results = run(module, items, seed=a.seed)
    out = save_results(results, module)
    print(summary_text(results) + f"; saved {out}")
    if a.write_d19:
        try:
            write_d19(results, module, out, lock=a.lock, dry_run=a.dry_run)
        except dec.Refused as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
