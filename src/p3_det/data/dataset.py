"""P3-04: TBX11K detector dataset: lung crop on the fly, C06 boxes, resize, box-safe augmentation.

Per sample:
  1. read the C04 512x512 PNG (CLAHE variant per D08) and the C05 crop for the D09 margin;
  2. take C06 ``orig_bbox_512`` boxes, shift into the crop, clip, drop boxes fully outside;
  3. resize crop -> input_size x input_size and scale the boxes;
  4. (train) box-safe augmentation, re-seeded per (seed, epoch, index) for reproducibility.

Which images: the requested C01 split only. TB images with boxes are positives; non-TB images
(tb_label 0) are negatives with an empty box list. TB images *without* boxes (e.g. uncertain TB)
are excluded: they contain disease, so they are not valid negatives, and they have no boxes.

Labels: C06 uses category_id 1 ("tb_lesion"); targets use label 0 for a single-class detector
(D-FINE num_classes = 1).
"""
from __future__ import annotations

import csv
import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .augment import AugmentConfig, augment
from .config import INPUT_SIZES, DataPaths, check_split, resolve_preprocessing
from .geometry import check_crop_box, clip_boxes, full512_to_crop, full512_to_det, xywh_to_xyxy

LABEL = 0


@dataclass(frozen=True)
class DetDataConfig:
    split: str
    input_size: int = 512
    margin: float | None = None        # None: locked D09, else PROVISIONAL 0.10
    clahe: bool | None = None          # None: locked D08, else PROVISIONAL off
    augment: bool | None = None        # None: on for train only
    aug: AugmentConfig = AugmentConfig()
    seed: int = 0
    min_box_size: float = 2.0          # px in detector space
    dataset: str = "tbx11k"
    eval_mode: bool = False            # keep TB images whose boxes are too small to train on (they are GT)


@dataclass
class Sample:
    image_id: str
    img_path: Path
    crop_box: tuple[int, int, int, int]
    boxes_full512: np.ndarray            # [k, 4] xyxy, as annotated (before crop)
    tb_label: int
    subtypes: list[str] = field(default_factory=list)


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def crop_and_resize(img512: np.ndarray, crop_box, input_size: int) -> np.ndarray:
    """Crop the 512 image to ``crop_box`` (end-exclusive) and resize to input_size^2 (bilinear)."""
    x0, y0, x1, y1 = check_crop_box(crop_box)
    crop = Image.fromarray(np.ascontiguousarray(img512[y0:y1, x0:x1]))
    return np.asarray(crop.resize((input_size, input_size), Image.Resampling.BILINEAR))


class TBDetectionDataset(Dataset):
    def __init__(self, cfg: DetDataConfig, paths: DataPaths | None = None, decisions_dir: Path | None = None):
        check_split(cfg.split, cfg.dataset)                   # before any file is opened
        if cfg.input_size not in INPUT_SIZES:
            raise ValueError(f"input_size must be one of {INPUT_SIZES}")
        self.cfg = cfg
        self.paths = paths or DataPaths.default()
        self.margin, self.clahe, self.provisional = resolve_preprocessing(cfg.margin, cfg.clahe, decisions_dir)
        self.do_augment = (cfg.split == "train") if cfg.augment is None else cfg.augment
        self.epoch = 0
        self.stats: Counter = Counter()
        self.samples = self._build_index()

    # --- index -------------------------------------------------------------------------------
    def _build_index(self) -> list[Sample]:
        cfg, p = self.cfg, self.paths
        with open(p.splits, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == cfg.split]
        manifest = {r["image_id"]: r for r in _read_parquet(p.manifest) if r["dataset"] == cfg.dataset}
        roi = {r["image_id"]: r for r in _read_parquet(p.lung_roi) if abs(r["margin"] - self.margin) < 1e-6}
        coco = json.loads(p.coco(cfg.split, self.margin).read_text())
        coco_img = {im["id"]: im for im in coco["images"]}
        anns: dict[str, list[dict]] = {}
        for a in coco["annotations"]:
            anns.setdefault(coco_img[a["image_id"]]["file_name"], []).append(a)
        coco_crop = {im["file_name"]: im.get("crop_box") for im in coco["images"]}

        samples = []
        for r in rows:
            iid, tb, has_boxes = r["image_id"], int(r["tb_label"]), r["has_boxes"] == "True"
            if tb == -1:
                self.stats["skipped_label_withheld"] += 1
                continue
            if tb == 1 and not has_boxes:
                self.stats["skipped_tb_without_boxes"] += 1
                continue
            m = manifest.get(iid)
            if m is None or m["decode_status"] != "ok":
                self.stats["skipped_unreadable_or_missing"] += 1
                continue
            if iid not in roi:
                raise KeyError(f"{iid}: no C05 row for margin {self.margin}")
            ro = roi[iid]
            crop = check_crop_box((ro["crop_x0"], ro["crop_y0"], ro["crop_x1"], ro["crop_y1"]))
            if coco_crop.get(iid) is not None and tuple(coco_crop[iid]) != crop:
                raise ValueError(f"{iid}: C06 crop_box {coco_crop[iid]} != C05 crop {crop}")
            a = anns.get(iid, [])
            if tb == 1 and not a:
                # every box may lie fully outside the crop (C06 outside_crop); then it is not usable
                self.stats["skipped_tb_all_boxes_outside_crop"] += 1
                continue
            if tb == 0 and a:
                raise ValueError(f"{iid}: non-TB image has C06 boxes")
            boxes = xywh_to_xyxy([x["orig_bbox_512"] for x in a]) if a else np.zeros((0, 4))
            self._check_c06_bbox(iid, a, boxes, crop)
            det, _ = full512_to_det(boxes, crop, cfg.input_size, min_size=0.0)
            if tb == 1 and not cfg.eval_mode and not len(
                    clip_boxes(det, cfg.input_size, cfg.input_size, cfg.min_box_size)[0]):
                # would silently become a false negative
                self.stats["skipped_tb_boxes_too_small_after_crop"] += 1
                continue
            rel =m["path_img512_clahe"] if self.clahe else m["path_img512"]
            samples.append(Sample(iid, self._resolve(rel), crop, boxes, tb, [x.get("tb_subtype") for x in a]))
            self.stats["positives" if tb else "negatives"] += 1
        return samples

    def _resolve(self, rel: str) -> Path:
        q = Path(rel)
        return q if q.is_absolute() else self.paths.root / q

    def _check_c06_bbox(self, iid: str, anns: list[dict], boxes_full: np.ndarray, crop) -> None:
        """Our crop-space boxes must match C06's own ``bbox`` (crop coords) within 1 px."""
        if not anns:
            return
        ours, keep = full512_to_crop(boxes_full, crop, min_size=0.0)
        theirs = xywh_to_xyxy([x["bbox"] for x in anns])[keep]
        if ours.shape != theirs.shape or np.abs(ours - theirs).max() > 1.0:
            raise ValueError(f"{iid}: C06 bbox disagrees with orig_bbox_512 shifted into the crop")

    # --- access ------------------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def load(self, index: int, augment_: bool | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (image uint8 [S,S], boxes float [k,4] in detector space, info)."""
        s = self.samples[index]
        S = self.cfg.input_size
        img = np.asarray(Image.open(s.img_path).convert("L"))
        if img.shape != (512, 512):
            raise ValueError(f"{s.image_id}: C04 image must be 512x512, got {img.shape}")
        img_det = crop_and_resize(img, s.crop_box, S)
        boxes, _ = full512_to_det(s.boxes_full512, s.crop_box, S, min_size=0.0)
        boxes, _ = clip_boxes(boxes, S, S, min_size=self.cfg.min_box_size)
        info = {"image_id": s.image_id, "crop_box": s.crop_box, "input_size": S,
                "n_boxes_annotated": len(s.boxes_full512), "aug": None}
        if self.do_augment if augment_ is None else augment_:
            rng = np.random.default_rng([self.cfg.seed, self.epoch, index])
            img_det, boxes, info["aug"] = augment(img_det, boxes, rng, self.cfg.aug)
        return img_det, boxes.astype(np.float32), info

    def __getitem__(self, index: int):
        img, boxes, info = self.load(index)
        image = torch.from_numpy(np.array(img, copy=True)).float().div_(255.0).unsqueeze(0).expand(3, -1, -1)
        target = {
            "boxes": torch.from_numpy(boxes).reshape(-1, 4),
            "labels": torch.full((len(boxes),), LABEL, dtype=torch.int64),
            "image_id": info["image_id"],
            "crop_box": torch.tensor(info["crop_box"], dtype=torch.int64),
            "input_size": info["input_size"],
        }
        return image.contiguous(), target

    def describe(self) -> dict:
        return {"split": self.cfg.split, "input_size": self.cfg.input_size, "margin": self.margin,
                "clahe": self.clahe, "provisional": self.provisional, "augment": self.do_augment,
                "seed": self.cfg.seed, "n": len(self), **self.stats}


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def make_dataloader(cfg: DetDataConfig, batch_size: int, num_workers: int = 2, shuffle: bool | None = None,
                    paths: DataPaths | None = None, decisions_dir: Path | None = None) -> DataLoader:
    """Train: shuffled, drop_last. Other splits: in order, all samples. Shuffle seeded by cfg.seed."""
    ds = TBDetectionDataset(cfg, paths, decisions_dir)
    if ds.provisional:
        warnings.warn(f"detector data is PROVISIONAL: {ds.describe()}", stacklevel=2)
    train = cfg.split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=train if shuffle is None else shuffle,
                      drop_last=train, num_workers=num_workers, collate_fn=collate,
                      generator=torch.Generator().manual_seed(cfg.seed), worker_init_fn=_worker_init)
