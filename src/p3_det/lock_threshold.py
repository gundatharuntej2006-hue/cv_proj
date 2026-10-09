"""P3-07: decision D18, detector confidence threshold (pre-registered rule, guide acknowledgement).

  1. python -m p3_det.lock_threshold criterion        -> COMMIT decisions/p3_det_conf_threshold.json
  2. python -m p3_det.lock_threshold result --run artifacts/p3/det/runs/<D17 run folder>   -> commit
  3. (only after the guide approved)  python -m p3_det.lock_threshold ack-guide --date YYYY-MM-DD --note "..."
  4. python -m p3_det.lock_threshold result --run <same> --lock                             -> commit

The threshold is computed on the development detections of the D17 checkpoint
(dev_detections_best.json of the D17 run, epoch checked against D17). Only development non-TB
images (no annotated boxes) enter the rule; TB images are used only to report sensitivity.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from . import decisions as dec
from .decisions import Refused
from .lock_checkpoint import D17_PATH, load_gt
from .metrics import detections_from_c10, image_level_lesion_sensitivity, wilson_ci

REPO = dec.REPO
D18_PATH = REPO / "decisions" / "p3_det_conf_threshold.json"
MAX_SHARE = 0.05
STEP = 0.001
FLOOR = 0.01           # C10 keeps boxes >= 0.01; lower thresholds are meaningless

RULE = ("lowest confidence threshold such that at most 5% of development non-TB images have any detection box "
        "above it")
CRITERION = (
    f"Pre-registered rule: \"{RULE}\". Operationalisation: computed on the development detections of the D17 "
    "checkpoint (EMA weights); non-TB images = development images without annotated lesion boxes; an image "
    "'has a box above t' if its highest box score s satisfies s >= t (the same comparison as "
    "n_lesions_above_thr in C10); candidate thresholds are the grid 0.010, 0.011, ..., 1.000; the threshold is "
    "the smallest grid value t with (non-TB images having a box with s >= t) / (all development non-TB images) "
    "<= 0.05. Reported with it: that share with a 95% Wilson CI and n, and TB image-level lesion sensitivity at t "
    "(Wilson CI). Requires the guide's acknowledgement (guide_ack) before locking."
)


def criterion_template(today: str | None = None) -> dict:
    return dec.new_decision("D18", "Detector confidence threshold", CRITERION,
                            [f"{FLOOR:.3f}..1.000 step {STEP}"], ["development"], today)


def grid() -> np.ndarray:
    return np.round(np.arange(FLOOR, 1.0 + STEP / 2, STEP), 3)


def lowest_threshold(max_scores, max_share: float = MAX_SHARE) -> tuple[float, int]:
    """Smallest grid t with #(max_score >= t) <= max_share * n. Returns (t, images with a box at t).

    ``max_scores`` holds, per non-TB image, its highest detection score (0 if it has no box).
    """
    m = np.asarray(max_scores, dtype=np.float64)
    if not len(m):
        raise Refused("no development non-TB images: the rule is undefined")
    allowed = int(np.floor(max_share * len(m) + 1e-9))
    g = grid()
    counts = (m[None, :] >= g[:, None]).sum(axis=1)       # non-increasing in t
    ok = np.flatnonzero(counts <= allowed)
    if not len(ok):
        raise Refused(f"no threshold up to 1.000 keeps non-TB images with a box <= {max_share:.0%}")
    return float(g[ok[0]]), int(counts[ok[0]])


def result(run_dir: Path, *, lock: bool = False, dry_run: bool = False, path: Path = D18_PATH,
           d17_path: Path = D17_PATH, today: str | None = None) -> dict:
    doc = dec.load_for_result(path, CRITERION)
    sha_rule = dec.criterion_commit(path, CRITERION)
    if sha_rule is None and not dry_run:
        raise Refused("the D18 rule is not committed yet: commit decisions/p3_det_conf_threshold.json first")

    run_dir = Path(run_dir)
    d17 = json.loads(d17_path.read_text()) if d17_path.exists() else {}
    det_doc = json.loads((run_dir / "dev_detections_best.json").read_text())
    if not dry_run:
        if not d17.get("locked"):
            raise Refused("D17 (checkpoint) must be locked before D18")
        if d17["value"]["run_dir"] != json.loads((run_dir / "run_meta.json").read_text())["run_dir"]:
            raise Refused("this run is not the D17 run")
        if det_doc.get("epoch") != d17["value"]["epoch"]:
            raise Refused(f"detections are from epoch {det_doc.get('epoch')}, D17 chose {d17['value']['epoch']}")

    gt = load_gt(run_dir)
    dets = detections_from_c10(det_doc)
    non_tb = sorted(i for i, g in gt.items() if not g.is_tb)
    maxima = [float(dets[i][:, 4].max()) if i in dets and len(dets[i]) else 0.0 for i in non_tb]
    t, n_hit = lowest_threshold(maxima)
    lo, hi = wilson_ci(n_hit, len(non_tb))
    sens = image_level_lesion_sensitivity(dets, gt, t)
    doc["value"] = {
        "conf_threshold": t,
        "rule": {"max_share_non_tb_with_box": MAX_SHARE, "comparison": "score >= t", "grid_step": STEP,
                 "grid_floor": FLOOR},
        "non_tb_share_with_box": {"value": round(n_hit / len(non_tb), 5), "ci_low": round(lo, 5),
                                  "ci_high": round(hi, 5), "ci_method": "wilson", "n_non_tb_images": len(non_tb),
                                  "n_with_box": n_hit},
        "tb_image_level_lesion_sensitivity": {k: (round(v, 5) if isinstance(v, float) else v)
                                              for k, v in sens.as_dict().items()},
        "detector": {"d17_epoch": det_doc.get("epoch"), "run_dir": run_dir.name,
                     "checkpoint_sha256": (d17.get("value") or {}).get("checkpoint", {}).get("sha256")},
        "criterion_commit": sha_rule,
        **({"guide_ack_record": doc["value"]["guide_ack_record"]} if "guide_ack_record" in doc.get("value", {}) else {}),
    }
    doc["evidence"] = [dec.repo_rel(run_dir / "dev_detections_best.json"),
                       dec.repo_rel(run_dir / "dev_ground_truth.json"), dec.repo_rel(d17_path)]
    if lock:
        if dry_run or sha_rule is None:
            raise Refused("cannot lock: rule commit not found (or dry run)")
        if not doc.get("guide_ack"):
            raise Refused("cannot lock D18 without the guide's acknowledgement (run ack-guide after approval)")
        acked = doc["value"].get("guide_ack_record", {}).get("conf_threshold_acknowledged")
        if acked != t:
            raise Refused(f"the guide acknowledged threshold {acked}, but the result is now {t}; ask again")
        doc.update(locked=True, locked_at=today or date.today().isoformat(), git_sha=dec.head_sha(path))
    dec.save(path, doc)
    return doc


def ack_guide(ack_date: str, note: str, path: Path = D18_PATH) -> dict:
    """Record the guide's acknowledgement. Run ONLY after the guide has actually approved."""
    doc = dec.load_for_result(path, CRITERION)
    if not doc.get("value"):
        raise Refused("compute the threshold first; the guide acknowledges a concrete result")
    if not note.strip():
        raise Refused("give a note (e.g. how/when the guide approved)")
    date.fromisoformat(ack_date)
    doc["guide_ack"] = True
    doc["value"]["guide_ack_record"] = {"date": ack_date, "note": note.strip(),
                                        "conf_threshold_acknowledged": doc["value"]["conf_threshold"]}
    dec.save(path, doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("criterion")
    r = sub.add_parser("result")
    r.add_argument("--run", required=True, type=Path)
    r.add_argument("--lock", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="skip commit/D17 checks; never locks")
    g = sub.add_parser("ack-guide", help="record the guide's approval (only after it happened)")
    g.add_argument("--date", required=True)
    g.add_argument("--note", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "criterion":
            dec.write_criterion(D18_PATH, criterion_template())
            print(f"wrote {D18_PATH.relative_to(REPO)}; commit it BEFORE running `result`")
            return 0
        if a.cmd == "ack-guide":
            ack_guide(a.date, a.note)
            print("guide_ack recorded")
            return 0
        doc = result(a.run, lock=a.lock, dry_run=a.dry_run)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    v = doc["value"]
    s, t = v["non_tb_share_with_box"], v["tb_image_level_lesion_sensitivity"]
    print(f"D18 threshold {v['conf_threshold']:.3f}: non-TB with a box {s['n_with_box']}/{s['n_non_tb_images']} = "
          f"{s['value']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]; TB image-level lesion sensitivity "
          f"{t['value']:.3f} [{t['ci_low']:.3f}, {t['ci_high']:.3f}] n_TB={t['n_tb_images']}; "
          f"guide_ack={doc['guide_ack']} locked={doc['locked']}")
    if not doc["guide_ack"]:
        print("REMINDER: D18 needs the guide's acknowledgement before it can be locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
