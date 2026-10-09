"""P3-02: crash-safe checkpoints with resume (model, optimiser, scheduler, epoch, RNG).

Layout of a checkpoint directory:
    ckpt_epoch0007.pt   one file per completed epoch (0-based epoch index); the last ``keep_last`` are kept
    ckpt_best.pt        copy of the best epoch, kept until a better one arrives
    best.json           {"epoch": 5, "metrics": {...}} sidecar for the best checkpoint

Every file is written to a temporary name in the same directory, fsynced, then renamed with
``os.replace``, so a kill mid-write leaves at worst a stray ``.tmp`` file that is ignored.
Resume also falls back to the previous checkpoint if the newest one cannot be loaded.

Usage in a training loop::

    state = resume_if_exists(ckpt_dir, model, optimizer, scheduler)
    for epoch in range(state.start_epoch, epochs):
        ...train one epoch...
        save_checkpoint(model, optimizer, scheduler, epoch, {"val_loss": v}, seed, ckpt_dir, is_best=...)
"""
from __future__ import annotations

import json
import os
import random
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1
BEST_NAME = "ckpt_best.pt"
BEST_JSON = "best.json"
_EPOCH_RE = re.compile(r"^ckpt_epoch(\d{4,})\.pt$")


def checkpoint_name(epoch: int) -> str:
    return f"ckpt_epoch{epoch:04d}.pt"


# --- RNG state ---------------------------------------------------------------------------------
# NumPy's state is stored as plain lists so checkpoints load with torch.load(weights_only=True).

def get_rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: dict) -> None:
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    kind, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        cuda = state["cuda"][: torch.cuda.device_count()]
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])


# --- atomic writes -----------------------------------------------------------------------------

def _atomic_write(path: Path, write) -> None:
    """Call ``write(fileobj)`` on a temp file next to ``path``, fsync, then rename over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _state(obj) -> Any:
    return obj.state_dict() if obj is not None else None


# --- public API --------------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch: int, metrics: dict, seed: int,
                    path: str | os.PathLike, *, is_best: bool = False, keep_last: int = 2,
                    extra: dict | None = None) -> Path:
    """Save the state after completed ``epoch`` (0-based) into directory ``path``.

    ``extra`` maps names to objects with ``state_dict()`` (e.g. a GradScaler or EMA) or to plain
    values; they are restored by ``resume_if_exists(..., extra=...)``. Returns the checkpoint path.
    """
    ckpt_dir = Path(path)
    payload = {
        "format_version": FORMAT_VERSION,
        "epoch": int(epoch),
        "seed": int(seed),
        "metrics": dict(metrics or {}),
        "model": _state(model),
        "optimizer": _state(optimizer),
        "scheduler": _state(scheduler),
        "extra": {k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in (extra or {}).items()},
        "rng": get_rng_state(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    target = ckpt_dir / checkpoint_name(epoch)
    _atomic_write(target, lambda f: torch.save(payload, f))
    if is_best:
        _atomic_write(ckpt_dir / BEST_NAME, lambda f: torch.save(payload, f))
        best = json.dumps({"epoch": int(epoch), "metrics": payload["metrics"]}, indent=1).encode()
        _atomic_write(ckpt_dir / BEST_JSON, lambda f: f.write(best))
    prune_checkpoints(ckpt_dir, keep_last=keep_last)
    return target


def list_checkpoints(ckpt_dir: str | os.PathLike) -> list[tuple[int, Path]]:
    """(epoch, path) of complete-looking epoch checkpoints, newest first. Temp files are ignored."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        return []
    found = []
    for p in d.iterdir():
        m = _EPOCH_RE.match(p.name)
        if m and p.is_file() and p.stat().st_size > 0:
            found.append((int(m.group(1)), p))
    return sorted(found, reverse=True)


def find_latest_checkpoint(ckpt_dir: str | os.PathLike) -> Path | None:
    found = list_checkpoints(ckpt_dir)
    return found[0][1] if found else None


def prune_checkpoints(ckpt_dir: str | os.PathLike, keep_last: int = 2) -> list[Path]:
    """Delete all but the newest ``keep_last`` epoch checkpoints (best is separate), plus stale temps."""
    d = Path(ckpt_dir)
    removed = [p for _, p in list_checkpoints(d)[keep_last:]]
    removed += [p for p in d.glob(".ckpt_*.tmp")]
    for p in removed:
        p.unlink(missing_ok=True)
    return removed


def load_checkpoint(path: str | os.PathLike, weights_only: bool = True) -> dict:
    return torch.load(path, map_location="cpu", weights_only=weights_only)


@dataclass
class ResumeState:
    start_epoch: int = 0                 # first epoch to run
    resumed_from: Path | None = None     # checkpoint used, None on a fresh start
    last_epoch: int | None = None        # last completed epoch
    seed: int | None = None
    metrics: dict = field(default_factory=dict)
    best: dict | None = None             # contents of best.json
    skipped: list[Path] = field(default_factory=list)  # unreadable checkpoints that were skipped


def read_best(ckpt_dir: str | os.PathLike) -> dict | None:
    p = Path(ckpt_dir) / BEST_JSON
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def resume_if_exists(ckpt_dir: str | os.PathLike, model, optimizer=None, scheduler=None, *,
                     extra: dict | None = None, restore_rng: bool = True,
                     weights_only: bool = True) -> ResumeState:
    """Restore the newest loadable checkpoint into the given objects; fresh start if none.

    Restores model, optimiser, scheduler, ``extra`` objects and (by default) Python/NumPy/torch/CUDA
    RNG state, so a resumed run continues as if it had not been interrupted.
    """
    state = ResumeState(best=read_best(ckpt_dir))
    for epoch, path in list_checkpoints(ckpt_dir):
        try:
            ckpt = load_checkpoint(path, weights_only=weights_only)
        except Exception as e:  # truncated / corrupt file: fall back to the previous one
            warnings.warn(f"skipping unreadable checkpoint {path.name}: {type(e).__name__}: {e}")
            state.skipped.append(path)
            continue
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        for name, obj in (extra or {}).items():
            if name in ckpt.get("extra", {}) and hasattr(obj, "load_state_dict"):
                obj.load_state_dict(ckpt["extra"][name])
        if restore_rng and "rng" in ckpt:
            set_rng_state(ckpt["rng"])
        state.start_epoch = ckpt["epoch"] + 1
        state.last_epoch = ckpt["epoch"]
        state.resumed_from = path
        state.seed = ckpt.get("seed")
        state.metrics = ckpt.get("metrics", {})
        return state
    return state
