"""Report paired development AP50 for completed provisional detector runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .decisions import PREPROCESSING_ABLATION, Refused
from .metrics import ap50, ap50_difference
from .select_size import load_run

SHARED_INPUTS = ("data/processed/v1/manifest.parquet", "data/boxes/v1/coco_development_full.json")
FIXED_SETTINGS = ("input_size", "schedule_hash", "seed", "dfine_commit", "repo_git_sha", "init", "amp")
SIMPLICITY_ORDER = ((None, False), (None, True), (0.10, False), (0.10, True))
MIN_AP50_GAIN = 0.01


def _load_candidate(path: Path) -> tuple[dict, tuple[float | None, bool], dict[str, str]]:
    run = load_run(path, expected_purpose=PREPROCESSING_ABLATION)
    meta, gt, detections = run["meta"], run["gt_doc"], run["det_doc"]
    prep = meta["preprocessing"]
    if prep.get("provisional") is not True or not isinstance(prep.get("clahe"), bool) or "margin" not in prep:
        raise Refused(f"{path.name}: expected explicit provisional preprocessing")
    if gt.get("split") != "development" or gt.get("coords") != "full512":
        raise Refused(f"{path.name}: expected full-image development ground truth")
    if detections.get("model_id") != meta["run_dir"] or detections.get("input_size") != meta["input_size"]:
        raise Refused(f"{path.name}: detections don't match the recorded run")
    if len(detections["images"]) != len(run["dets"]) or run["dets"].keys() != run["gt"].keys():
        raise Refused(f"{path.name}: detections must cover each development image exactly once")
    if not all(np.isfinite(v).all() for v in run["dets"].values()) or not all(
            np.isfinite(v.boxes).all() for v in run["gt"].values()):
        raise Refused(f"{path.name}: non-finite detections or ground truth")
    hashes = meta["input_sha256"]
    split_key = "c01_split" if "c01_split" in hashes else "data/splits/tbx11k_splits_v1.csv"
    shared = {key: hashes[key] for key in (split_key, *SHARED_INPUTS) if hashes.get(key)}
    if len(shared) != 3:
        raise Refused(f"{path.name}: missing split, manifest, or original development-box digest")
    return run, (prep["margin"], prep["clahe"]), shared


def compare(baseline: Path, candidates: list[Path], *, resamples: int, seed: int) -> dict:
    if len(candidates) != len(SIMPLICITY_ORDER) - 1:
        raise Refused("provide the baseline and all three other protocol candidates")
    if resamples < 1 or seed < 0:
        raise ValueError("provide a positive bootstrap count and a non-negative seed")
    reference, reference_prep, shared = _load_candidate(Path(baseline))
    if reference_prep != SIMPLICITY_ORDER[0]:
        raise Refused("baseline must use full images with CLAHE off")
    if reference["meta"]["input_size"] != 512:
        raise Refused("the detector preprocessing study uses the 512 configuration")
    gt = reference["gt"]
    precision = reference["meta"].get("development_artifact_precision", "rounded_3dp")
    if not any(image.is_tb for image in gt.values()):
        raise Refused("AP50 requires at least one annotated development lesion")
    accepted = {reference_prep: reference}
    for path in map(Path, candidates):
        run, prep, inputs = _load_candidate(path)
        if prep not in SIMPLICITY_ORDER:
            raise Refused(f"{path.name}: preprocessing is outside the study matrix")
        if prep in accepted:
            raise Refused(f"{path.name}: duplicate preprocessing candidate")
        if inputs != shared or run["gt_doc"]["images"] != reference["gt_doc"]["images"]:
            raise Refused(f"{path.name}: different development ground truth or shared input digests")
        for key in FIXED_SETTINGS:
            if run["meta"][key] != reference["meta"][key]:
                raise Refused(f"{path.name}: differs from baseline in {key}")
        if run["det_doc"]["min_score"] != reference["det_doc"]["min_score"]:
            raise Refused(f"{path.name}: differs from baseline in detection score floor")
        if run["meta"].get("development_artifact_precision", "rounded_3dp") != precision:
            raise Refused(f"{path.name}: differs from baseline in development artifact precision")
        accepted[prep] = run

    results = []
    comparisons = []
    current = reference
    for settings in SIMPLICITY_ORDER:
        run = accepted[settings]
        meta = run["meta"]
        train_counts = {key: count for key, count in meta.get("train_data", {}).items()
                        if key in ("n", "positives", "negatives") or key.startswith("skipped_")}
        data = {"run": meta["run_dir"], "preprocessing": meta["preprocessing"], "train_counts": train_counts,
                "ap50": ap50(run["dets"], gt, B=resamples, seed=seed).as_dict()}
        results.append(data)
        if settings == SIMPLICITY_ORDER[0]:
            continue
        delta = ap50_difference(current["dets"], run["dets"], gt, B=resamples, seed=seed)
        # Allow for floating-point error when an exact 0.01 is computed by subtraction.
        meets_margin = delta.value >= MIN_AP50_GAIN or math.isclose(
            delta.value, MIN_AP50_GAIN, rel_tol=0, abs_tol=1e-12)
        promoted = meets_margin and delta.ci_low > 0
        comparisons.append({"current": current["meta"]["run_dir"], "candidate": meta["run_dir"],
                            "paired_ap50_difference": delta.as_dict(), "promoted": promoted})
        if promoted:
            current = run
    return {"status": "provisional", "split": "development", "coords": "full512",
            "development_artifact_precision": precision,
            "baseline": reference["meta"]["run_dir"], "shared_input_sha256": shared,
            "c01_verified_at_training": "c01_split" in shared,
            "bootstrap": {"resamples": resamples, "seed": seed, "confidence": 0.95,
                          "paired": True, "strata": ["tb", "non_tb"]},
            "minimum_ap50_gain": MIN_AP50_GAIN, "runs": results, "comparisons": comparisons,
            "choice_for_review": current["meta"]["run_dir"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--resamples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, help="New JSON file (default: print to stdout)")
    args = parser.parse_args(argv)
    try:
        report = json.dumps(compare(args.baseline, args.candidate, resamples=args.resamples, seed=args.seed),
                            indent=2, allow_nan=False)
        if args.out is None:
            print(report)
        else:
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(report + "\n")
    except (OSError, ValueError, KeyError, TypeError, Refused) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
