"""P3-07: decision D17, detector training length / checkpoint, by development AP50 only.

  1. python -m p3_det.lock_checkpoint criterion          -> commit decisions/p3_det_checkpoint.json
  2. python -m p3_det.lock_checkpoint result --run artifacts/p3/det/runs/<run folder>
         [--checkpoint path/to/ckpt_best.pt] [--lock]

<run folder> is the P3-06 run of the D16-chosen input size on LOCKED preprocessing, copied from Drive
(run_meta.json, train_log.csv, best.json, dev_ground_truth.json, dev_detections_best.json,
checkpoints_sha256.txt). The chosen epoch is the one with the highest development AP50 in
train_log.csv (earliest on ties); it must agree with best.json and dev_detections_best.json.
The checkpoint SHA-256 comes from checkpoints_sha256.txt and, if --checkpoint is given, is
recomputed from the file and must match.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from . import decisions as dec
from .decisions import Refused
from .metrics import ImageGT, ap50, bootstrap_settings, detections_from_c10

REPO = dec.REPO
D16_PATH = REPO / "decisions" / "p3_det_input_size.json"
D17_PATH = REPO / "decisions" / "p3_det_checkpoint.json"
EVIDENCE_DIR = REPO / "artifacts" / "p3" / "det" / "lock"

CRITERION = (
    "The D-FINE detector checkpoint (training length) is the epoch with the highest development AP at IoU 0.50 "
    "(p3_det.metrics.ap50 on the EMA weights, logged after every epoch in train_log.csv) within the fixed P3-06 "
    "schedule of the D16-chosen input size, trained on the LOCKED D08/D09 preprocessing; ties go to the earliest "
    "epoch. Only train (training) and development (selection) are used. The checkpoint is that epoch's "
    "ckpt_best.pt, identified by its SHA-256; inference uses its EMA weights. Its development AP50 is reported "
    "with a 95% percentile bootstrap CI over images, stratified TB/non-TB (B and seed from D21)."
)


def criterion_template(today: str | None = None) -> dict:
    return dec.new_decision("D17", "Detector training length / checkpoint", CRITERION,
                            ["every epoch of the D16-chosen run"], ["train", "development"], today)


def read_sha_file(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            h, name = line.split(maxsplit=1)
            out[name.strip()] = h
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def best_epoch(log: list[dict]) -> tuple[int, float]:
    """Highest dev_ap50, earliest epoch on ties."""
    best = max(log, key=lambda r: (float(r["dev_ap50"]), -int(r["epoch"])))
    return int(best["epoch"]), float(best["dev_ap50"])


def load_gt(run_dir: Path) -> dict[str, ImageGT]:
    doc = json.loads((run_dir / "dev_ground_truth.json").read_text())
    return {k: ImageGT(v["boxes"], v["subtype"]) for k, v in doc["images"].items()}


def result(run_dir: Path, *, checkpoint: Path | None = None, lock: bool = False, dry_run: bool = False,
           path: Path = D17_PATH, d16_path: Path = D16_PATH, evidence_dir: Path = EVIDENCE_DIR,
           today: str | None = None) -> dict:
    doc = dec.load_for_result(path, CRITERION)
    sha_rule = dec.criterion_commit(path, CRITERION)
    if sha_rule is None and not dry_run:
        raise Refused("the D17 criterion is not committed yet: commit decisions/p3_det_checkpoint.json first")

    run_dir = Path(run_dir)
    need = ["run_meta.json", "train_log.csv", "best.json", "dev_ground_truth.json", "dev_detections_best.json",
            "checkpoints_sha256.txt"]
    missing = [n for n in need if not (run_dir / n).exists()]
    if missing:
        raise Refused(f"{run_dir}: missing {missing}")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    with open(run_dir / "train_log.csv", newline="") as f:
        log = list(csv.DictReader(f))
    epochs = int(meta["config"]["schedule"]["epochs"])
    if [int(r["epoch"]) for r in log] != list(range(epochs)):
        raise Refused(f"{run_dir.name}: train_log.csv does not cover epochs 0..{epochs - 1} (run not finished)")

    d16 = json.loads(d16_path.read_text()) if d16_path.exists() else {}
    if not dry_run:
        if not d16.get("locked"):
            raise Refused("D16 (input size) must be locked before D17")
        if d16["value"]["input_size"] != meta["input_size"]:
            raise Refused(f"run is {meta['input_size']} px but D16 chose {d16['value']['input_size']}")
        if d16["value"]["preprocessing"] != meta["preprocessing"]:
            raise Refused("run preprocessing differs from the D16 runs")
    if meta["preprocessing"].get("provisional") and lock:
        raise Refused("PROVISIONAL preprocessing: cannot lock D17")

    epoch, ap_logged = best_epoch(log)
    best_json = json.loads((run_dir / "best.json").read_text())
    if best_json["epoch"] != epoch:
        raise Refused(f"best.json says epoch {best_json['epoch']} but train_log.csv's best is epoch {epoch}")
    det_doc = json.loads((run_dir / "dev_detections_best.json").read_text())
    if det_doc.get("epoch") != epoch:
        raise Refused(f"dev_detections_best.json is from epoch {det_doc.get('epoch')}, not {epoch}")

    shas = read_sha_file(run_dir / "checkpoints_sha256.txt")
    if "ckpt_best.pt" not in shas:
        raise Refused("checkpoints_sha256.txt has no entry for ckpt_best.pt")
    sha = shas["ckpt_best.pt"]
    if checkpoint is not None and sha256_file(checkpoint) != sha:
        raise Refused(f"{checkpoint} does not match the recorded SHA-256 of ckpt_best.pt")

    B, seed, source = bootstrap_settings()
    r = ap50(detections_from_c10(det_doc), load_gt(run_dir), B=B, seed=seed)
    if abs(r.value - ap_logged) > 1e-4:
        raise Refused(f"recomputed AP50 {r.value:.5f} != logged {ap_logged:.5f} for epoch {epoch}")

    ckpt_info = {"file": "ckpt_best.pt", "sha256": sha, "verified_from_file": checkpoint is not None,
                 "location": f"Drive: tb_checkpoints/{meta['run_dir']}/ckpt_best.pt",
                 "inference_weights": "extra['ema']['module'] (EMA)"}
    ap = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.as_dict().items()}
    evidence_dir.mkdir(parents=True, exist_ok=True)
    ev = evidence_dir / "d17_evidence.json"
    ev.write_text(json.dumps({
        "run_dir": meta["run_dir"], "input_size": meta["input_size"], "epoch": epoch,
        "ap50_by_epoch": [{"epoch": int(x["epoch"]), "dev_ap50": float(x["dev_ap50"])} for x in log],
        "dev_ap50_at_epoch": ap, "checkpoint": ckpt_info, "bootstrap": {"B": B, "seed": seed, "source": source},
    }, indent=2) + "\n")
    doc["value"] = {
        "epoch": epoch, "training_length_epochs": epoch + 1, "schedule_epochs": epochs,
        "input_size": meta["input_size"], "run_dir": meta["run_dir"], "seed": meta["seed"],
        "schedule_hash": meta["schedule_hash"], "preprocessing": meta["preprocessing"],
        "dev_ap50": {k: ap[k] for k in ("value", "ci_low", "ci_high", "n_images", "n_tb_images",
                                        "n_non_tb_images", "n_boxes")},
        "checkpoint": ckpt_info, "criterion_commit": sha_rule,
    }
    doc["evidence"] = [dec.repo_rel(ev)] + [dec.repo_rel(run_dir / n) for n in
                                            ("train_log.csv", "checkpoints_sha256.txt", "run_meta.json")]
    if lock:
        if dry_run or sha_rule is None:
            raise Refused("cannot lock: criterion commit not found (or dry run)")
        doc.update(locked=True, locked_at=today or date.today().isoformat(), git_sha=dec.head_sha(path))
    dec.save(path, doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("criterion")
    r = sub.add_parser("result")
    r.add_argument("--run", required=True, type=Path)
    r.add_argument("--checkpoint", type=Path)
    r.add_argument("--lock", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="skip commit/D16 checks; never locks")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "criterion":
            dec.write_criterion(D17_PATH, criterion_template())
            print(f"wrote {D17_PATH.relative_to(REPO)}; commit it before running `result`")
            return 0
        doc = result(a.run, checkpoint=a.checkpoint, lock=a.lock, dry_run=a.dry_run)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    v = doc["value"]
    print(f"D17: epoch {v['epoch']} (training length {v['training_length_epochs']} of {v['schedule_epochs']}), "
          f"dev AP50 {v['dev_ap50']['value']:.4f} [{v['dev_ap50']['ci_low']:.4f}, {v['dev_ap50']['ci_high']:.4f}] "
          f"n={v['dev_ap50']['n_images']}; ckpt_best.pt sha256 {v['checkpoint']['sha256']}; locked={doc['locked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
