# -*- coding: utf-8 -*-
"""Shared, memory-bounded loaders for the scanner run.

Host RAM available during this run was 0.33-0.98 GB of 16.4 GB, while 5.32 GB of VRAM was
free. The regional caches are 1.35 GB (TBX11K) and 555 MB (Vietnam) on disk, so loading
them with np.load would swap or fail.

`stream_regions` therefore reads the `reg` member straight out of the npz zip in row
blocks, normalises each block on the GPU, and keeps only the result in VRAM. Host memory
never holds more than one block (~56 MB at the default block size).

Equivalence to the committed pipeline, which is what matters:
  scripts/p2_32_mil_sweep.py::repool(X, 12, 12) has f == 1, so it performs no pooling. It
  casts to float32, divides each 768-d region vector by its L2 norm clamped at 1e-9, and
  casts back to float16. `_normalise` below does exactly that, on the GPU, block by block.
  `verify_equivalence.py` checks this numerically against the committed CPU implementation
  on a subset, and the end-to-end check is that the unprojected arm reproduces the
  published Vietnam TUNE AUROC of 0.7237.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REG = ROOT / "artifacts/p2/regional"
TB_CATS = ("active_tb", "latent_tb", "active_latent_tb")
SEEDS = (17, 29, 43, 61, 79)
GRID = 12
REGIONS = GRID * GRID


def npy_header(zf, member):
    with zf.open(member) as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    return tuple(shape), dtype, bool(fortran)


def _normalise(block_t):
    """float16 [m, R, C] on device -> L2-normalised float16, matching repool(X, 12, 12)."""
    g = block_t.float()
    n = g.norm(dim=2, keepdim=True).clamp_min(1e-9)
    return (g / n).half()


def stream_regions(path, keep_idx=None, block=256, device="cuda", verbose=True):
    """Read `reg` from an npz in row blocks; return an L2-normalised float16 device tensor.

    keep_idx : sorted array of row indices to retain, or None for all. Rows outside it are
               read and discarded, so host memory stays bounded regardless of selectivity.
    """
    import torch

    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        shape, dtype, fortran = npy_header(zf, "reg.npy")
        if fortran:
            raise ValueError("fortran-ordered array not supported: %s" % path)
        n, regions, chan = shape
        if regions != REGIONS:
            raise ValueError("%s has %d regions, expected %d" % (path.name, regions, REGIONS))
        itemsize = np.dtype(dtype).itemsize
        row_bytes = regions * chan * itemsize

        if keep_idx is None:
            keep = np.arange(n)
        else:
            keep = np.asarray(sorted(keep_idx))
        keep_set = set(int(i) for i in keep)
        out = torch.empty((len(keep), regions, chan), dtype=torch.float16, device=device)

        pos = 0
        with zf.open("reg.npy") as f:
            np.lib.format.read_magic(f)
            np.lib.format._read_array_header(f, (1, 0))
            for s0 in range(0, n, block):
                m = min(block, n - s0)
                raw = f.read(m * row_bytes)
                if len(raw) != m * row_bytes:
                    raise IOError("short read at row %d of %s" % (s0, path.name))
                take = [i for i in range(s0, s0 + m) if i in keep_set]
                if not take:
                    continue
                arr = np.frombuffer(raw, dtype=dtype).reshape(m, regions, chan)
                sel = arr[[i - s0 for i in take]]
                t = torch.from_numpy(np.ascontiguousarray(sel)).to(device, non_blocking=True)
                out[pos:pos + len(take)] = _normalise(t)
                pos += len(take)
                del arr, sel, t
        if pos != len(keep):
            raise AssertionError("kept %d rows, expected %d" % (pos, len(keep)))
    if verbose:
        mb = out.numel() * 2 / 1e6
        print("  streamed %s -> %s on %s (%.0f MB VRAM)"
              % (path.name, tuple(out.shape), device, mb))
    return out, keep


def read_ids(path):
    with zipfile.ZipFile(path) as zf:
        with zf.open("image_id.npy") as f:
            return [str(x) for x in np.load(io.BytesIO(f.read()), allow_pickle=False)]


def agg(scores, how):
    """scores: [n, R] device tensor -> [n]. Mirrors p2_32_mil_sweep.agg_torch."""
    if how == "mean":
        return scores.mean(1)
    if how == "max":
        return scores.max(1).values
    if how == "top3":
        return scores.topk(min(3, scores.shape[1]), dim=1).values.mean(1)
    if how == "top5":
        return scores.topk(min(5, scores.shape[1]), dim=1).values.mean(1)
    if how == "lse":
        return scores.logsumexp(1)
    raise ValueError(how)


def train_head(xt, yt, xv, ydv, how="max", seed=17, epochs=120, lr=3e-3, bs=256, wd=1e-4,
               patience_max=12):
    """Device-resident copy of p2_32_mil_sweep.train.

    Same optimiser, same learning rate, weight decay, batch size, epoch budget, logit
    adjustment and early-stopping rule. The only difference is that the features already
    live on the device instead of being copied there inside the function.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    dev = xt.device
    ytt = torch.as_tensor(yt, dtype=torch.float32, device=dev)
    lin = torch.nn.Linear(xt.shape[2], 1).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    prior = float(ytt.mean())
    adjust = float(np.log(max(prior, 1e-6) / max(1 - prior, 1e-6)))
    n = len(ytt)
    best, best_state, patience = -1.0, None, 0
    for _ in range(epochs):
        lin.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            z = agg(lin(xt[k].float()).squeeze(-1), how)
            torch.nn.functional.binary_cross_entropy_with_logits(z - adjust, ytt[k]).backward()
            opt.step()
        lin.eval()
        with torch.no_grad():
            sv = torch.cat([agg(lin(xv[i:i + 512].float()).squeeze(-1), how)
                            for i in range(0, len(xv), 512)]).float().cpu().numpy()
        a = roc_auc_score(ydv, sv)
        if a > best + 1e-5:
            best, patience = a, 0
            best_state = {k2: v.detach().clone() for k2, v in lin.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_max:
                break
    lin.load_state_dict(best_state)
    W = lin.weight.detach().clone()
    b = lin.bias.detach().clone()
    return W, b, float(best)


def score_head(x, W, b, how="max", chunk=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            s = (x[i:i + chunk].float() @ W.T).squeeze(-1) + b
            out.append(agg(s, how).float().cpu())
    return torch.cat(out).numpy()


def boot_ci(y, p, n_boot=4000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    vals = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        vals.append(roc_auc_score(y[k], p[k]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot(y, pa, pb, n_boot=4000, seed=0):
    """CI for AUROC(pb) - AUROC(pa) on identical cases."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pa = np.asarray(pa, float)
    pb = np.asarray(pb, float)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        d.append(roc_auc_score(y[k], pb[k]) - roc_auc_score(y[k], pa[k]))
    return (float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def vietnam_meta():
    """pid -> dict, plus the TUNE/REPORT membership sets. One image per patient."""
    import csv
    rows = list(csv.DictReader(io.open(
        ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv", encoding="utf-8-sig")))
    meta = {str(r["pid"]): r for r in rows}
    split = json.load(io.open(
        ROOT / "artifacts/p2/external/vietnam_tune_report_split.json", encoding="utf-8"))
    return meta, set(str(x) for x in split["tune"]), set(str(x) for x in split["report"])


def tbx11k_rows():
    """The exact training and development rows the committed pipeline uses."""
    import pandas as pd
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    return c01
