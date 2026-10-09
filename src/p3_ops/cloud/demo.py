"""P3-02: toy training loop that exercises the checkpoint helper (CPU or GPU, seconds per epoch).

Notebook / Kaggle deliberate-kill test:
    python -m p3_ops.cloud.demo --run killtest --epochs 20 --epoch-seconds 30
    # stop the session mid-run, start a new one, run the same command again: it resumes.
Simulated kill (what the tests do):
    python -m p3_ops.cloud.demo --run sim --epochs 6 --kill-at-epoch 3
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import read_best, resume_if_exists, save_checkpoint
from .sync import CheckpointStore


class SimulatedKill(RuntimeError):
    """Raised mid-epoch to imitate a session being killed."""


@dataclass
class DemoResult:
    model_state: dict
    start_epoch: int                      # first epoch run in this call (0 on a fresh start)
    epochs_completed: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    seed: int | None = None


def set_seed(seed: int) -> str:
    try:
        from tbcore.seed import set_all  # P4 utility
        set_all(seed)
        return "tbcore.seed.set_all"
    except ImportError:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return "fallback (tbcore.seed not available)"


def _toy_data(n: int = 256, seed: int = 1234):
    g = torch.Generator().manual_seed(seed)  # data fixed, independent of the training RNG
    x = torch.randn(n, 8, generator=g)
    y = x @ torch.randn(8, 1, generator=g) + 0.1 * torch.randn(n, 1, generator=g)
    return x[:192], y[:192], x[192:], y[192:]


def build(seed: int):
    set_seed(seed)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    return model, opt, sched


def train(ckpt_dir: str | Path, epochs: int = 6, steps_per_epoch: int = 6, seed: int = 0,
          kill_at: tuple[int, int] | None = None, epoch_seconds: float = 0.0,
          store: CheckpointStore | None = None, verbose: bool = False) -> DemoResult:
    """Train the toy model, resuming from ``ckpt_dir`` if it holds checkpoints.

    ``kill_at=(epoch, step)`` raises SimulatedKill at that step, after some of the epoch's work.
    """
    model, opt, sched = build(seed)
    state = resume_if_exists(ckpt_dir, model, opt, sched)
    if state.seed is not None and state.seed != seed:
        raise ValueError(f"checkpoint seed {state.seed} != requested seed {seed}")
    best = (state.best or {}).get("metrics", {}).get("val_loss", float("inf"))
    if verbose:
        print(f"start at epoch {state.start_epoch}"
              + (f" (resumed from {state.resumed_from.name})" if state.resumed_from else " (fresh)"))
    x, y, xv, yv = _toy_data()
    bs = len(x) // steps_per_epoch
    result = DemoResult(model_state={}, start_epoch=state.start_epoch, seed=seed)
    for epoch in range(state.start_epoch, epochs):
        t0 = time.monotonic()
        model.train()
        perm = torch.randperm(len(x))  # uses the (restored) global RNG
        for step in range(steps_per_epoch):
            if kill_at == (epoch, step):
                raise SimulatedKill(f"killed at epoch {epoch} step {step}")
            idx = perm[step * bs:(step + 1) * bs]
            loss = nn.functional.mse_loss(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch_seconds:
                time.sleep(epoch_seconds / steps_per_epoch)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(nn.functional.mse_loss(model(xv), yv))
        is_best = val < best
        best = min(best, val)
        save_checkpoint(model, opt, sched, epoch, {"val_loss": val}, seed, ckpt_dir, is_best=is_best)
        if store is not None:
            store.push()
        result.epochs_completed.append(epoch)
        result.val_losses.append(val)
        if verbose:
            print(f"epoch {epoch:3d}  val_loss {val:.5f}  {'best ' if is_best else ''}"
                  f"({time.monotonic() - t0:.1f}s)", flush=True)
    result.model_state = {k: v.clone() for k, v in model.state_dict().items()}
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="p3_ckpt_demo", help="run name (checkpoint sub-directory)")
    ap.add_argument("--ckpt-dir", help="explicit directory; default chosen by CheckpointStore")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epoch-seconds", type=float, default=0.0, help="slow epochs down for a manual kill")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kill-at-epoch", type=int, help="simulate a kill mid-way through this epoch")
    args = ap.parse_args(argv)

    store = None
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
    else:
        store = CheckpointStore.for_run(args.run)
        store.pull()
        ckpt_dir = store.dir
        print(f"env={store.env} dir={store.dir} remote={store.remote}")
    kill = (args.kill_at_epoch, 3) if args.kill_at_epoch is not None else None
    try:
        res = train(ckpt_dir, args.epochs, seed=args.seed, kill_at=kill,
                    epoch_seconds=args.epoch_seconds, store=store, verbose=True)
    except SimulatedKill as e:
        print(f"SIMULATED KILL: {e}. Run the same command again to resume.")
        return 1
    print(f"done: epochs {res.start_epoch}..{args.epochs - 1} run in this session; best {read_best(ckpt_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
