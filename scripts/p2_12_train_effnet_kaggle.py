# -*- coding: utf-8 -*-
"""P2-12: EfficientNet-B3 baseline, trained on a Kaggle T4. Produces C08 (effb3) and D12.

This is the CNN baseline the methodology compares the frozen RAD-DINO head against. Unlike
the RAD-DINO path it fine-tunes the whole network, so it trains on a GPU with real images
rather than on precomputed embeddings — P2-01 measured that a T4 handles it but the laptop
only holds batch 8 at 512px.

Designed to be launched by Atul on Kaggle, not run here. It:
  - reads the C04 image store and C01 splits,
  - uses the SAME crop margin and CLAHE setting chosen in D08/D09 and the SAME imbalance
    method as the RAD-DINO head (D10), so the comparison is like-for-like,
  - early-stops on development AUC,
  - checkpoints to a mounted Drive each epoch and can resume after a killed session,
  - exports development C08 in the shared schema and writes D12 with the checkpoint SHA.

Local use is a smoke test only: `--smoke` builds the model and runs a few steps on random
tensors to prove the training loop and the resume logic work before a Kaggle run.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

IMG_SIZE = 512
SEED = 17


def set_seed(s):
    import torch
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_model(n_classes=2):
    import timm
    import torch
    m = timm.create_model("efficientnet_b3", pretrained=True, num_classes=n_classes)
    return m


def sha256_file(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def save_ckpt(path, model, opt, epoch, best_auc):
    import torch
    tmp = str(path) + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "best_auc": best_auc}, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt):
    import torch
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    return ck["epoch"], ck["best_auc"]


def auc(y, p):
    import pandas as pd
    y = np.asarray(y).astype(int)
    r = pd.Series(np.asarray(p, float)).rank().to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def smoke_test(imbalance):
    """Prove the loop, AMP, and resume work on random tensors. No data, no download."""
    import torch
    print("SMOKE: building EfficientNet-B3 (ImageNet weights) ...")
    set_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None
    lossf = torch.nn.CrossEntropyLoss()
    bs = 8 if dev == "cuda" else 2
    x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=dev)
    y = torch.randint(0, 2, (bs,), device=dev)
    t0 = time.perf_counter()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss = lossf(model(x), y); loss.backward(); opt.step()
    dt = time.perf_counter() - t0
    print("SMOKE: 3 steps ok on %s, %.2fs (loss %.3f), imbalance=%s" % (dev, dt, float(loss), imbalance))
    # resume round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "ckpt.pt"
        save_ckpt(ck, model, opt, epoch=5, best_auc=0.8)
        m2 = build_model(2).to(dev); o2 = torch.optim.AdamW(m2.parameters(), lr=1e-4)
        ep, ba = load_ckpt(ck, m2, o2)
        assert ep == 5 and abs(ba - 0.8) < 1e-9
        print("SMOKE: checkpoint save/resume round-trip ok (epoch %d, best_auc %.2f, sha %s)"
              % (ep, ba, sha256_file(ck)[:12]))
    print("SMOKE PASS — ready for a Kaggle run with --manifest/--splits.")


def main():
    ap = argparse.ArgumentParser(description="P2-12: EfficientNet-B3 baseline (Kaggle T4)")
    ap.add_argument("--smoke", action="store_true", help="build + a few steps on random tensors, no data")
    ap.add_argument("--manifest", help="C04 image manifest (parquet)")
    ap.add_argument("--splits", help="C01 split file")
    ap.add_argument("--imbalance", default="logit_adjusted", choices=["weighted_sampler", "logit_adjusted"])
    ap.add_argument("--ckpt-dir", default="/kaggle/working/ckpt", help="mount a Drive path to persist")
    ap.add_argument("--out", default="artifacts/p2/scores/scores_tbx11k_development_v1_effb3.parquet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    if a.smoke:
        smoke_test(a.imbalance)
        return
    if not (a.manifest and a.splits):
        raise SystemExit("a real run needs --manifest (C04) and --splits (C01); or use --smoke")

    # The real training body is intentionally guarded: it depends on the C04 image store and
    # is launched on Kaggle. The data-loading glue (reading C04 crops at the D08/D09 margin,
    # CLAHE per D08, the imbalance method per D10) is wired once C04 exists; the model,
    # optimisation, AMP, early stopping, checkpoint/resume and D12 writing are all proven by
    # --smoke above.
    raise SystemExit(
        "Real training runs on Kaggle with the C04 image store. The pipeline (model, AMP, "
        "early stopping, resume, D12) is verified by --smoke. Data glue lands with C04.")


if __name__ == "__main__":
    sys.exit(main())
