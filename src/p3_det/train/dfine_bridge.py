"""P3-06: thin bridge to the pinned D-FINE submodule (third_party/D-FINE @ 956d170).

We use D-FINE's model, criterion, postprocessor, optimiser, LR/warm-up schedulers, EMA and AMP
scaler exactly as its YAML config builds them, but feed them our own data (P3-04 adapter) and
run our own loop, so evaluation (P3-05) and checkpoints (P3-02) are ours.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.geometry import det_to_full512
from .config import DFINE_DIR, RunConfig

NUM_CLASSES = 1   # single "tb_lesion" class, label 0 (P3-04 adapter)


def import_dfine():
    """Put D-FINE on sys.path and import its registry (its top-level package is called ``src``)."""
    if str(DFINE_DIR) not in sys.path:
        sys.path.insert(0, str(DFINE_DIR))
    import src as dfine_src  # noqa: F401  (registers modules)
    from src.core import YAMLConfig
    from src.data.dataloader import BatchImageCollateFunction
    return YAMLConfig, BatchImageCollateFunction


@dataclass
class Components:
    model: torch.nn.Module
    criterion: torch.nn.Module
    postprocessor: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: object
    warmup: object
    ema: object
    scaler: object
    clip_max_norm: float
    device: torch.device
    dfine_cfg: dict


def _yaml_config(cfg: RunConfig, amp: bool):
    YAMLConfig, _ = import_dfine()
    size = cfg.input_size
    overrides = {
        "num_classes": NUM_CLASSES,
        "eval_spatial_size": [size, size],
        "use_amp": bool(amp),
        # No ImageNet backbone download: with init=coco the COCO checkpoint supplies the backbone too;
        # init=none (smoke tests only) trains from scratch offline.
        "HGNetv2": {"pretrained": False},
        "epochs": cfg.schedule["epochs"],
    }
    with tempfile.TemporaryDirectory(prefix="p3_dfine_cfg_") as work:
        cfg_file = Path(work) / "run.yml"
        cfg_file.write_text(json.dumps({"__include__": [str(DFINE_DIR / cfg.model["dfine_config"])], **overrides}))
        return YAMLConfig(str(cfg_file))


def build_inference(cfg: RunConfig, device: str | torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """(model, postprocessor) only: no optimiser, loss or EMA. Weights must be loaded by the caller."""
    ycfg = _yaml_config(cfg, amp=False)
    dev = torch.device(device)
    return ycfg.model.to(dev).eval(), ycfg.postprocessor.to(dev).eval()


def build(cfg: RunConfig, device: str | torch.device, amp: bool) -> Components:
    s = cfg.schedule
    ycfg = _yaml_config(cfg, amp)
    dev = torch.device(device)
    model = ycfg.model.to(dev)
    optimizer = ycfg.optimizer                    # built from model.parameters()
    for pg in optimizer.param_groups:
        pg["lr"] *= float(s["lr_scale"])
        pg["initial_lr"] = pg["lr"]
    lr_scheduler = ycfg.lr_scheduler
    warmup = ycfg.lr_warmup_scheduler
    ema = ycfg.ema
    if ema is not None:
        ema.module.to(dev)
    return Components(
        model=model, criterion=ycfg.criterion.to(dev), postprocessor=ycfg.postprocessor.to(dev),
        optimizer=optimizer, lr_scheduler=lr_scheduler, warmup=warmup, ema=ema,
        scaler=ycfg.scaler if amp else None, clip_max_norm=float(ycfg.yaml_cfg.get("clip_max_norm", 0.0)),
        device=dev, dfine_cfg=ycfg.yaml_cfg)


# --- pretrained weights ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weights(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / url.rsplit("/", 1)[-1]
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def load_pretrained(model: torch.nn.Module, path: Path) -> dict:
    """Load D-FINE COCO weights where names and shapes match; report what was skipped.

    The 80-class COCO heads do not fit our 1-class heads, so they stay randomly initialised.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)  # official D-FINE release file
    state = state["ema"]["module"] if "ema" in state else state.get("model", state)
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    return {"loaded": len(matched), "skipped_shape": sorted(k for k in state if k in own and k not in matched),
            "missing": sorted(k for k in own if k not in state)}


# --- data glue ------------------------------------------------------------------------------------

class DFineTargets(torch.utils.data.Dataset):
    """Wrap the P3-04 dataset: xyxy pixel boxes -> D-FINE targets (cxcywh normalised, int labels)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        image, t = self.ds[i]
        S = float(t["input_size"])
        b = t["boxes"]
        cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0],
                              b[:, 3] - b[:, 1]], dim=1) / S if len(b) else torch.zeros((0, 4))
        return image, {"boxes": cxcywh.float(), "labels": t["labels"], "image_id": torch.tensor([i]),
                       "orig_size": torch.tensor([int(S), int(S)])}


def make_train_collate(cfg: RunConfig):
    """D-FINE's own batch collate (multi-scale until multiscale_stop_epoch)."""
    _, BatchImageCollateFunction = import_dfine()
    s = cfg.schedule
    stop = s["multiscale_stop_epoch"] if s["multiscale"] else 0
    return BatchImageCollateFunction(base_size=cfg.input_size, base_size_repeat=3 if s["multiscale"] else None,
                                     stop_epoch=stop)


# --- one epoch / prediction -----------------------------------------------------------------------

def train_one_epoch(c: Components, loader, epoch: int) -> dict:
    """Same step as D-FINE det_engine.train_one_epoch (AMP, grad clip, EMA, per-step warm-up)."""
    c.model.train()
    c.criterion.train()
    losses = []
    n_steps = len(loader)
    for step, (images, targets) in enumerate(loader):
        images = images.to(c.device)
        targets = [{k: v.to(c.device) for k, v in t.items()} for t in targets]
        metas = dict(epoch=epoch, step=step, global_step=epoch * n_steps + step, epoch_step=n_steps)
        if c.scaler is not None:
            with torch.autocast(device_type=c.device.type, cache_enabled=True):
                outputs = c.model(images, targets=targets)
            with torch.autocast(device_type=c.device.type, enabled=False):
                loss = sum(c.criterion(outputs, targets, **metas).values())
            c.scaler.scale(loss).backward()
            if c.clip_max_norm > 0:
                c.scaler.unscale_(c.optimizer)
                torch.nn.utils.clip_grad_norm_(c.model.parameters(), c.clip_max_norm)
            c.scaler.step(c.optimizer)
            c.scaler.update()
        else:
            outputs = c.model(images, targets=targets)
            loss = sum(c.criterion(outputs, targets, **metas).values())
            loss.backward()
            if c.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(c.model.parameters(), c.clip_max_norm)
            c.optimizer.step()
        c.optimizer.zero_grad()
        if c.ema is not None:
            c.ema.update(c.model)
        if c.warmup is not None:
            c.warmup.step()
        value = float(loss.detach())
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
        losses.append(value)
    if c.warmup is None or c.warmup.finished():
        c.lr_scheduler.step()
    return {"train_loss": float(np.mean(losses)) if losses else float("nan"), "steps": n_steps,
            "lr": float(c.optimizer.param_groups[0]["lr"])}


def to_tensor(img_det: np.ndarray) -> torch.Tensor:
    """uint8 [S,S] detector-space image -> float [3,S,S] in [0,1] (same as the P3-04 dataset)."""
    return torch.from_numpy(np.array(img_det, copy=True)).float().div_(255.0).unsqueeze(0).expand(3, -1, -1)


@torch.no_grad()
def infer(model: torch.nn.Module, postprocessor: torch.nn.Module, images_det: list[np.ndarray],
          crop_boxes: list, input_size: int, device: torch.device, amp: bool,
          min_score: float = 0.01) -> list[np.ndarray]:
    """THE inference path (training-time eval, R3 detect and C10 export all use it).

    images_det: uint8 [S,S] crops already resized to input_size. Returns, per image, [m, 5] rows
    [x0, y0, x1, y1, score] in full 512 coordinates, score >= min_score, highest score first.
    """
    S = input_size
    model.eval()
    images = torch.stack([to_tensor(i) for i in images_det]).to(device)
    with torch.autocast(device_type=device.type, enabled=amp):
        outputs = model(images)
    sizes = torch.tensor([[S, S]] * len(images_det), device=device)
    out = []
    for r, crop in zip(postprocessor(outputs, sizes), crop_boxes):
        keep = r["scores"] >= min_score
        boxes = r["boxes"][keep].float().cpu().numpy()
        scores = r["scores"][keep].float().cpu().numpy()
        full = det_to_full512(boxes, crop, S) if len(boxes) else np.zeros((0, 4))
        # drop boxes squashed onto the crop edge (< 0.01 px), so C10's 0.001 px rounding never empties one
        ok = (full[:, 2] - full[:, 0] >= 0.01) & (full[:, 3] - full[:, 1] >= 0.01)
        rows = np.concatenate([full[ok], scores[ok, None]], axis=1)
        out.append(rows[np.argsort(-rows[:, 4], kind="mergesort")])
    return out


def predict(c: Components, model: torch.nn.Module, dataset, batch_size: int, amp: bool,
            min_score: float = 0.01) -> dict[str, np.ndarray]:
    """Detections for every image of an (un-augmented) P3-04 dataset: {image_id: [m, 5]} in 512 coords."""
    S = dataset.cfg.input_size
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(dataset), batch_size):
        chunk = range(start, min(start + batch_size, len(dataset)))
        imgs = [dataset.load(i, augment_=False)[0] for i in chunk]
        rows = infer(model, c.postprocessor, imgs, [dataset.samples[i].crop_box for i in chunk], S,
                     c.device, amp, min_score)
        out.update({dataset.samples[i].image_id: r for i, r in zip(chunk, rows)})
    return out
