"""P3-06: train D-FINE (single class tb_lesion) on train, evaluate AP50 on development every epoch.

    python -m p3_det.train --config dfine_512.yaml            # Kaggle/Colab T4, auto-resume
    python -m p3_det.train --config dfine_512.yaml --smoke    # tiny CPU run on FAKE P1 data

Run directory (synced to Drive by the P3-02 helper), named
``{run_name}_m{margin}_clahe{0|1}[_PROVISIONAL]`` so provisional and locked runs never mix:
    run_meta.json            config, schedule hash, seed (+ which seed utility), preprocessing, versions
    train_log.csv            one row per epoch: loss, lr, development AP50 (+ n), time, peak memory
    dev_ground_truth.json    development GT used for AP (all original boxes, 512 coords)
    dev_detections_last.json development detections of the latest epoch (EMA weights, 512 coords)
    dev_detections_best.json same, best development AP50 epoch so far
    ckpt_epochNNNN.pt (last 2), ckpt_best.pt, best.json   (P3-02 checkpoints)

Splits: train for training, development for evaluation. Nothing else is ever loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from p3_ops.cloud import CheckpointStore, resume_if_exists, save_checkpoint
from p3_ops.seed import set_seed

from ..data import DataPaths, DetDataConfig, TBDetectionDataset
from ..metrics import ImageGT, ap50
from . import dfine_bridge as bridge
from .config import DFINE_DIR, REPO, RunConfig, load_run_config, require_trainable

TRAIN_SPLIT, EVAL_SPLIT = "train", "development"
CSV_FIELDS = ["epoch", "train_loss", "lr", "steps", "dev_ap50", "dev_n_images", "dev_n_tb_images",
              "dev_n_boxes", "epoch_seconds", "peak_mem_gb", "seed", "input_size", "margin", "clahe",
              "provisional", "schedule_hash"]
FORMAT = "p3_dev_detections_v1"


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def image_subtype(subtypes: list) -> str | None:
    s = {x for x in subtypes if x}
    if not s:
        return None
    if "active+latent" in s or s >= {"active", "latent"}:
        return "active+latent"
    return next(iter(s)) if len(s) == 1 else None


def dev_ground_truth(ds: TBDetectionDataset) -> dict[str, ImageGT]:
    """All development images of the adapter: TB images with their original boxes, non-TB with none."""
    return {s.image_id: ImageGT(s.boxes_full512, image_subtype(s.subtypes) if len(s.boxes_full512) else None)
            for s in ds.samples}


def detections_doc(run_name: str, input_size: int, epoch: int, dets: dict[str, np.ndarray]) -> dict:
    return {"format": FORMAT, "model_id": run_name, "input_size": input_size, "coords": "full512",
            "epoch": epoch, "min_score": 0.01,
            "images": [{"image_id": k, "boxes": np.round(v, 3).tolist()} for k, v in sorted(dets.items())]}


def _rewrite_csv(path: Path, keep_before_epoch: int) -> None:
    """On resume, drop log rows of epochs that will be re-run (they were never checkpointed)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < keep_before_epoch]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run_dir_name(cfg: RunConfig, margin: float, clahe: bool, provisional: bool) -> str:
    return f"{cfg.run_name}_m{margin:.2f}_clahe{int(clahe)}" + ("_PROVISIONAL" if provisional else "")


def train(config: str | Path, *, device: str | None = None, data_root: Path | None = None,
          ckpt_root: Path | None = None, weights_cache: Path | None = None,
          max_epochs: int | None = None, decisions_dir: Path | None = None) -> Path:
    cfg = load_run_config(config)
    require_trainable(cfg)                     # epochs / batch_size must be set (P3-01)
    s = cfg.schedule
    seed_util = set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(s["amp"]) and device.startswith("cuda")
    paths = DataPaths(data_root) if data_root else DataPaths.default()

    common = dict(input_size=cfg.input_size, margin=cfg.data["margin"], clahe=cfg.data["clahe"], seed=cfg.seed)
    train_ds = TBDetectionDataset(DetDataConfig(TRAIN_SPLIT, aug=cfg.augment, **common), paths, decisions_dir)
    dev_ds = TBDetectionDataset(DetDataConfig(EVAL_SPLIT, augment=False, eval_mode=True, **common), paths,
                                decisions_dir)
    prep = {"margin": train_ds.margin, "clahe": train_ds.clahe, "provisional": train_ds.provisional}

    name = run_dir_name(cfg, **prep)
    if ckpt_root:
        store = CheckpointStore(dir=Path(ckpt_root) / name, env="local")
        store.dir.mkdir(parents=True, exist_ok=True)
    else:
        store = CheckpointStore.for_run(name)
        store.pull()
    rd = store.dir
    print(f"[p3-06] run dir {rd}  device={device} amp={amp}  {prep}", flush=True)

    meta_path = rd / "run_meta.json"
    meta = {
        "run_name": cfg.run_name, "run_dir": name, "input_size": cfg.input_size, "config": cfg.raw,
        "config_file": cfg.path.name, "schedule_hash": cfg.schedule_hash, "seed": cfg.seed, "seed_util": seed_util,
        "preprocessing": prep, "splits": {"train": TRAIN_SPLIT, "eval": EVAL_SPLIT},
        "train_data": train_ds.describe(), "dev_data": dev_ds.describe(),
        "dfine_commit": _git_sha(DFINE_DIR), "repo_git_sha": _git_sha(REPO),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(), "device": device, "amp": amp,
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for k in ("schedule_hash", "input_size", "seed", "preprocessing"):
            if old[k] != meta[k]:
                raise RuntimeError(f"existing run in {rd} has a different {k} ({old[k]} != {meta[k]}); "
                                   "use a new run_name instead of resuming")
        meta = {**old, "resumed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    else:
        meta["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    gt = dev_ground_truth(dev_ds)
    _atomic_json(rd / "dev_ground_truth.json", {
        "split": EVAL_SPLIT, "coords": "full512", "preprocessing": prep,
        "images": {k: {"boxes": v.boxes.round(3).tolist(), "subtype": v.subtype} for k, v in sorted(gt.items())},
        "excluded": {k: v for k, v in dev_ds.stats.items() if k.startswith("skipped")}})

    c = bridge.build(cfg, device, amp)
    extra = {k: v for k, v in (("ema", c.ema), ("scaler", c.scaler), ("warmup", c.warmup)) if v is not None}
    state = resume_if_exists(rd, c.model, c.optimizer, c.lr_scheduler, extra=extra)
    if state.resumed_from is None and cfg.model["init"] == "coco":
        wpath = bridge.fetch_weights(cfg.model["init_url"], weights_cache or Path(tempfile.gettempdir()) / "p3_weights")
        report = bridge.load_pretrained(c.model, wpath)
        if c.ema is not None:
            c.ema.module.load_state_dict(c.model.state_dict())
        meta["init"] = {"url": cfg.model["init_url"], "sha256": bridge.sha256_file(wpath),
                        "loaded_tensors": report["loaded"], "skipped_shape": report["skipped_shape"],
                        "missing": len(report["missing"])}
        print(f"[p3-06] COCO init: {report['loaded']} tensors loaded, heads re-initialised: "
              f"{report['skipped_shape']}", flush=True)
    elif state.resumed_from is None:
        meta["init"] = {"url": None, "note": "random init (smoke/test only)"}
    _atomic_json(meta_path, meta)
    _rewrite_csv(rd / "train_log.csv", state.start_epoch)

    collate = bridge.make_train_collate(cfg)
    loader = torch.utils.data.DataLoader(
        bridge.DFineTargets(train_ds), batch_size=s["batch_size"], shuffle=True, drop_last=True,
        num_workers=s["num_workers"], collate_fn=collate, generator=torch.Generator().manual_seed(cfg.seed),
        persistent_workers=False)
    best = (state.best or {}).get("metrics", {}).get("dev_ap50", -1.0)
    last_epoch = s["epochs"] if max_epochs is None else min(s["epochs"], max_epochs)

    for epoch in range(state.start_epoch, last_epoch):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        collate.set_epoch(epoch)
        loader.generator.manual_seed(cfg.seed * 1000 + epoch)      # resumable shuffling
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = bridge.train_one_epoch(c, loader, epoch)
        eval_model = c.ema.module if c.ema is not None else c.model
        dets = bridge.predict(c, eval_model, dev_ds, s["eval_batch_size"], amp, cfg.eval["min_score"])
        r = ap50(dets, gt, ci=False)
        is_best = r.value > best
        best = max(best, r.value)
        doc = detections_doc(cfg.run_name, cfg.input_size, epoch, dets)
        _atomic_json(rd / "dev_detections_last.json", doc)
        if is_best:
            _atomic_json(rd / "dev_detections_best.json", doc)
        _append_csv(rd / "train_log.csv", {
            "epoch": epoch, **stats, "dev_ap50": round(r.value, 5), "dev_n_images": r.n_images,
            "dev_n_tb_images": r.n_tb_images, "dev_n_boxes": r.n_boxes,
            "epoch_seconds": round(time.time() - t0, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3) if torch.cuda.is_available() else "",
            "seed": cfg.seed, "input_size": cfg.input_size, **prep, "schedule_hash": cfg.schedule_hash})
        save_checkpoint(c.model, c.optimizer, c.lr_scheduler, epoch,
                        {"dev_ap50": r.value, "train_loss": stats["train_loss"]}, cfg.seed, rd,
                        is_best=is_best, extra=extra)
        store.push()
        print(f"[p3-06] epoch {epoch}/{s['epochs'] - 1} loss {stats['train_loss']:.4f} "
              f"dev AP50 {r.value:.4f} (n={r.n_images}, boxes={r.n_boxes}) {time.time() - t0:.0f}s"
              + ("  best" if is_best else ""), flush=True)
    return rd


def main(argv=None) -> int:
    import sys
    import warnings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="dfine_512.yaml / dfine_640.yaml (src/p3_det/configs)")
    ap.add_argument("--device")
    ap.add_argument("--smoke", action="store_true", help="CPU, FAKE P1 data, D-FINE-N, 2 epochs, no download")
    args = ap.parse_args(argv)
    warnings.filterwarnings("default", message="PROVISIONAL")
    if not args.smoke:
        train(args.config, device=args.device)
        return 0
    import yaml
    from ..data.fake_p1 import make_fake_p1_tree
    work = Path(tempfile.mkdtemp(prefix="p3_06_smoke_"))
    cfg = load_run_config(args.config).raw
    cfg["model"].update(variant="n", dfine_config="configs/dfine/custom/dfine_hgnetv2_n_custom.yml", init="none")
    cfg["schedule"].update(epochs=2, batch_size=2, eval_batch_size=4, num_workers=0, amp=False,
                           multiscale_stop_epoch=1)
    smoke_cfg = work / Path(args.config).name
    smoke_cfg.write_text(yaml.safe_dump(cfg))
    rd = train(smoke_cfg, device="cpu", data_root=make_fake_p1_tree(work / "p1", n_per_split=6),
               ckpt_root=work / "ckpt", decisions_dir=work / "no_decisions")
    print(f"[p3-06] smoke run ok: {rd}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
