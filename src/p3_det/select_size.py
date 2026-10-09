"""P3-06: decision D16, detector input size 512 vs 640 by development AP50.

Two steps, in this order (the gate checks that the criterion predates the evidence):

  1. python -m p3_det.select_size criterion
       writes decisions/p3_det_input_size.json with the pre-registered criterion and
       preregistered_at = today, value = {} and locked = false.  -> COMMIT THIS FILE NOW.
  2. python -m p3_det.select_size compare --run512 <dir> --run640 <dir> [--lock]
       checks the criterion is unchanged and committed, checks both runs are comparable
       (same schedule hash, seed, preprocessing; both finished; same development GT), computes
       AP50 + 95% bootstrap CI for each and the paired difference, applies the rule, runs the
       contingency check, and fills value/evidence. --lock locks D16 only when the runs used
       LOCKED D08/D09 preprocessing and the contingency did not trigger.

Run directories are the P3-06 training output folders (copied from Drive), containing
run_meta.json, train_log.csv, dev_ground_truth.json and dev_detections_last.json.

Exit codes: 0 done, 2 refused (inputs or criterion not valid), 3 CONTINGENCY (stop; tell P4).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from . import decisions as dec
from .decisions import Refused
from .metrics import ImageGT, ap50, ap50_difference, bootstrap_settings, detections_from_c10

REPO = Path(__file__).resolve().parents[2]
D16_PATH = REPO / "decisions" / "p3_det_input_size.json"
EVIDENCE_DIR = REPO / "artifacts" / "p3" / "det" / "select_size"
CONTINGENCY_AP50 = 0.05

CRITERION = (
    "The D-FINE TB-lesion detector input size is chosen between 512 and 640 by development AP at IoU 0.50 "
    "(methodology section 7), computed with p3_det.metrics.ap50 (pycocotools-equivalent, full 512-image "
    "coordinates, all original ground-truth boxes, D21) on each run's FINAL-epoch EMA weights. Both runs use "
    "configs identical except input size (same schedule hash: seed, epochs, batch size, learning rates, "
    "augmentation) and the LOCKED D08/D09 preprocessing; runs on provisional preprocessing are not eligible. "
    "The size with the higher point estimate is kept; an exact tie keeps 512 (native TBX11K resolution). "
    "95% CIs: percentile bootstrap over development images, stratified TB/non-TB, B and seed from D21 "
    "(default B=2000, seed 0); the paired bootstrap CI of AP50(640) - AP50(512) is reported and does not "
    "change the rule. Contingency: if the chosen size has development AP50 < 0.05, D16 is not locked, tuning "
    "stops, and the detector contingency is raised with P4 (negative result reported, link rule disabled, "
    "Grad-CAM as the main localisation evidence)."
)


# --- step 1: criterion -----------------------------------------------------------------------------

def write_criterion(path: Path = D16_PATH, today: str | None = None) -> dict:
    return dec.write_criterion(path, dec.new_decision(
        "D16", "Detector input size (512 vs 640)", CRITERION, [512, 640], ["train", "development"], today))


def criterion_commit(path: Path = D16_PATH) -> str | None:
    return dec.criterion_commit(path, CRITERION)


# --- step 2: compare -------------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    need = ["run_meta.json", "train_log.csv", "dev_ground_truth.json", "dev_detections_last.json"]
    missing = [n for n in need if not (run_dir / n).exists()]
    if missing:
        raise Refused(f"{run_dir}: missing {missing}")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    with open(run_dir / "train_log.csv", newline="") as f:
        log = list(csv.DictReader(f))
    gt_doc = json.loads((run_dir / "dev_ground_truth.json").read_text())
    det_doc = json.loads((run_dir / "dev_detections_last.json").read_text())
    epochs = int(meta["config"]["schedule"]["epochs"])
    if det_doc.get("epoch") != epochs - 1 or [int(r["epoch"]) for r in log] != list(range(epochs)):
        raise Refused(f"{run_dir.name}: run not finished (needs epochs 0..{epochs - 1} in the log and "
                      f"final-epoch detections; detections are from epoch {det_doc.get('epoch')})")
    gt = {k: ImageGT(v["boxes"], v["subtype"]) for k, v in gt_doc["images"].items()}
    return {"dir": run_dir, "meta": meta, "log": log, "gt": gt, "gt_doc": gt_doc,
            "dets": detections_from_c10(det_doc)}


def check_comparable(r512: dict, r640: dict) -> None:
    m1, m2 = r512["meta"], r640["meta"]
    if (m1["input_size"], m2["input_size"]) != (512, 640):
        raise Refused(f"expected a 512 run and a 640 run, got {m1['input_size']} and {m2['input_size']}")
    for k in ("schedule_hash", "seed", "preprocessing"):
        if m1[k] != m2[k]:
            raise Refused(f"runs differ in {k}: {m1[k]} vs {m2[k]} (identical schedule and seed required)")
    if r512["gt_doc"]["images"] != r640["gt_doc"]["images"]:
        raise Refused("the two runs were evaluated on different development ground truth")


def _res(r) -> dict:
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.as_dict().items()}


def compare(run512: Path, run640: Path, *, lock: bool = False, allow_uncommitted: bool = False,
            path: Path = D16_PATH, evidence_dir: Path = EVIDENCE_DIR, today: str | None = None) -> tuple[dict, int]:
    doc = dec.load_for_result(path, CRITERION)
    sha = criterion_commit(path)
    if sha is None and not allow_uncommitted:
        raise Refused("the criterion is not committed yet: commit decisions/p3_det_input_size.json, then compare")

    r512, r640 = load_run(run512), load_run(run640)
    check_comparable(r512, r640)
    B, seed, source = bootstrap_settings()
    gt = r512["gt"]
    a512 = ap50(r512["dets"], gt, B=B, seed=seed)
    a640 = ap50(r640["dets"], gt, B=B, seed=seed)
    diff = ap50_difference(r512["dets"], r640["dets"], gt, B=B, seed=seed)
    chosen = 640 if a640.value > a512.value else 512
    best = max(a512.value, a640.value)
    prep = r512["meta"]["preprocessing"]
    contingency = bool(np.isnan(best) or best < CONTINGENCY_AP50)

    evidence_dir.mkdir(parents=True, exist_ok=True)
    ev_path = evidence_dir / "d16_comparison.json"
    comparison = {
        "ap50_512": _res(a512), "ap50_640": _res(a640), "ap50_640_minus_512": _res(diff),
        "bootstrap": {"B": B, "seed": seed, "source": source},
        "runs": {str(r["meta"]["input_size"]): {
            "run_dir": r["meta"]["run_dir"], "epochs": r["meta"]["config"]["schedule"]["epochs"],
            "final_epoch_log": r["log"][-1], "init": r["meta"].get("init"),
            "dfine_commit": r["meta"].get("dfine_commit"), "seed_util": r["meta"].get("seed_util")}
            for r in (r512, r640)},
        "schedule_hash": r512["meta"]["schedule_hash"], "seed": r512["meta"]["seed"], "preprocessing": prep,
    }
    ev_path.write_text(json.dumps(comparison, indent=2) + "\n")

    doc["value"] = {
        "input_size": chosen,
        "ap50_512": {k: comparison["ap50_512"][k] for k in ("value", "ci_low", "ci_high", "n_images",
                                                           "n_tb_images", "n_non_tb_images", "n_boxes")},
        "ap50_640": {k: comparison["ap50_640"][k] for k in ("value", "ci_low", "ci_high", "n_images",
                                                           "n_tb_images", "n_non_tb_images", "n_boxes")},
        "ap50_640_minus_512": {k: comparison["ap50_640_minus_512"][k] for k in ("value", "ci_low", "ci_high")},
        "seed": r512["meta"]["seed"], "schedule_hash": r512["meta"]["schedule_hash"],
        "preprocessing": prep, "criterion_commit": sha,
        "contingency": {"threshold_ap50": CONTINGENCY_AP50, "triggered": contingency},
    }
    doc["evidence"] = [dec.repo_rel(ev_path)] + [dec.repo_rel(Path(r["dir"]) / n) for r in (r512, r640)
                                              for n in ("train_log.csv", "run_meta.json")]

    code = 0
    if contingency:
        code = 3
    elif lock:
        if prep.get("provisional"):
            raise Refused("runs used PROVISIONAL preprocessing; re-run both sizes on locked D08/D09 before locking")
        if sha is None:
            raise Refused("cannot lock: criterion commit not found")
        doc.update(locked=True, locked_at=today or date.today().isoformat(), git_sha=dec.head_sha(path))
    dec.save(path, doc)
    return doc, code


def _report(doc: dict, code: int) -> None:
    v = doc["value"]
    f = lambda r: f"{r['value']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"  # noqa: E731
    print(f"AP50 512: {f(v['ap50_512'])}  n_images={v['ap50_512']['n_images']} "
          f"(TB {v['ap50_512']['n_tb_images']}, non-TB {v['ap50_512']['n_non_tb_images']}), boxes={v['ap50_512']['n_boxes']}")
    print(f"AP50 640: {f(v['ap50_640'])}")
    print(f"640 - 512: {f(v['ap50_640_minus_512'])} (paired bootstrap)")
    print(f"chosen input size: {v['input_size']}  preprocessing: {v['preprocessing']}  locked: {doc['locked']}")
    if code == 3:
        print("\n*** CONTINGENCY: best development AP50 < 0.05. STOP: do not keep tuning. D16 not locked. ***\n"
              "Raise the detector contingency with P4: report the negative result honestly, disable the\n"
              "detector link rule, and make Grad-CAM the main localisation evidence.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("criterion", help="write the pre-registered criterion (then commit it)")
    c = sub.add_parser("compare", help="compare two finished runs and fill D16")
    c.add_argument("--run512", required=True, type=Path)
    c.add_argument("--run640", required=True, type=Path)
    c.add_argument("--lock", action="store_true")
    c.add_argument("--allow-uncommitted", action="store_true", help="dry runs only; never with --lock")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "criterion":
            write_criterion()
            print(f"wrote {D16_PATH}. Commit it now, before running `compare`:\n"
                  f"  git add {D16_PATH.relative_to(REPO)} && git commit -m \"P3-06: pre-register D16 criterion\"")
            return 0
        if a.lock and a.allow_uncommitted:
            raise Refused("--lock cannot be combined with --allow-uncommitted")
        doc, code = compare(a.run512, a.run640, lock=a.lock, allow_uncommitted=a.allow_uncommitted)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    _report(doc, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
