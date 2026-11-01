# -*- coding: utf-8 -*-
"""Zambia scoring on Kaggle - FAST version. Replaces p2_zambia_kaggle.py.

The first version was badly engineered for this environment and could run for hours:
the lung segmenter was never moved to the GPU, and it did three backbone passes per
image when only one of the three needs segmentation at all.

This version:
  * drops the lung masking entirely (it is the only part that needs the segmenter);
  * scores the two maskings that need no segmentation - full image and border-only;
  * batches properly and prints progress every 100 images so it is obvious it is alive.

Expected runtime on a T4: a few minutes. If it is slower than that, the accelerator is
off - check Settings -> Accelerator -> GPU.

Lungs-only can be added later in a second run if the first two numbers justify it.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

IN = "/kaggle/input"
OUT = "/kaggle/working"
BORDER_FRAC = 0.12
BATCH = 32

print("GPU available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ONLY")
if not torch.cuda.is_available():
    print("\n*** No GPU. Settings -> Accelerator -> GPU T4, then Run All again. ***\n")

# ---------------------------------------------------------------- what is actually here
dcm = sorted(glob.glob(os.path.join(IN, "**", "*.dcm"), recursive=True))
print("DICOM files:", len(dcm))
if not dcm:
    # the images may be under another extension or inside archives; show the layout
    allf = glob.glob(os.path.join(IN, "**", "*"), recursive=True)
    allf = [p for p in allf if os.path.isfile(p)]
    print("no .dcm found; %d files total. extensions present:" % len(allf))
    ext = {}
    for p in allf:
        e = os.path.splitext(p)[1].lower() or "(none)"
        ext[e] = ext.get(e, 0) + 1
    for e, n in sorted(ext.items(), key=lambda kv: -kv[1])[:15]:
        print("   %-10s %d" % (e, n))
    print("\nfirst 20 paths:")
    for p in allf[:20]:
        print("   ", p.replace(IN, ""))

for p in glob.glob(os.path.join(IN, "**", "*.csv"), recursive=True)[:8]:
    try:
        d = pd.read_csv(p, nrows=3)
        print("\n%s  (%d cols)\n   %s" % (p.replace(IN, ""), d.shape[1], list(d.columns)[:30]))
    except Exception as e:
        print("could not read", p, e)

# ---------------------------------------------------------------- image -> 512
def dicom_to_512(path):
    """Identical normalisation to the other four sites. Do not change in isolation."""
    import cv2
    import pydicom
    from pydicom.pixels import apply_voi_lut
    d = pydicom.dcmread(path)
    if d.get("VOILUTFunction", "") == "":
        d.VOILUTFunction = "LINEAR"
    try:
        a = apply_voi_lut(d.pixel_array, d).astype(np.float32)
    except Exception:
        a = d.pixel_array.astype(np.float32)
    if d.get("PhotometricInterpretation") == "MONOCHROME1":
        a = a.max() - a
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    return cv2.resize(a.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)


def mask_border(img):
    out = img.astype(np.float32).copy()
    h, w = out.shape
    bh, bw = int(round(h * BORDER_FRAC)), int(round(w * BORDER_FRAC))
    keep = np.zeros(out.shape, bool)
    keep[:bh, :] = keep[-bh:, :] = True
    keep[:, :bw] = keep[:, -bw:] = True
    out[~keep] = float(out.mean())
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- backbone
from transformers import AutoImageProcessor, AutoModel

MODEL_ID, REVISION = "microsoft/rad-dino", "110cbc18d5133582e320b43d53bf5c44e410c936"
proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
dev = "cuda" if torch.cuda.is_available() else "cpu"
bb = AutoModel.from_pretrained(MODEL_ID, revision=REVISION)
bb = (bb.half() if dev == "cuda" else bb.float()).eval().to(dev)
for prm in bb.parameters():
    prm.requires_grad_(False)
print("backbone on", dev)


def embed(images):
    rgb = [np.stack([im] * 3, axis=-1) for im in images]
    x = proc(images=rgb, return_tensors="pt")["pixel_values"]
    x = x.to(dev, dtype=torch.float16 if dev == "cuda" else torch.float32)
    with torch.inference_mode():
        o = bb(pixel_values=x)
        f = o.pooler_output if getattr(o, "pooler_output", None) is not None \
            else o.last_hidden_state[:, 0]
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return f.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- heads
def load_head(name):
    hits = glob.glob(os.path.join(IN, "**", name), recursive=True)
    if not hits:
        return None
    with np.load(hits[0]) as z:
        return z["W"].astype(np.float32), z["b"].astype(np.float32)


HEADS = {k: load_head(f) for k, f in
         (("full", "head.npz"), ("border_only", "head_border_only.npz"))}
HEADS = {k: v for k, v in HEADS.items() if v is not None}
print("heads loaded:", list(HEADS))
if not HEADS:
    raise SystemExit("upload head.npz (and head_border_only.npz) as a Kaggle input")


def p_tb(emb, Wb):
    W, b = Wb
    lg = emb @ W.T + b
    e = np.exp(lg - lg.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


# ---------------------------------------------------------------- score
ids, out = [], {k: [] for k in HEADS}
buf_full, buf_border, buf_id = [], [], []
t0, skipped = time.time(), 0


def flush():
    if not buf_id:
        return
    if "full" in HEADS:
        out["full"].extend(p_tb(embed(buf_full), HEADS["full"]).tolist())
    if "border_only" in HEADS:
        out["border_only"].extend(p_tb(embed(buf_border), HEADS["border_only"]).tolist())
    ids.extend(buf_id)
    buf_full.clear()
    buf_border.clear()
    buf_id.clear()


for i, path in enumerate(dcm):
    try:
        img = dicom_to_512(path)
    except Exception:
        skipped += 1
        continue
    buf_full.append(img)
    if "border_only" in HEADS:
        buf_border.append(mask_border(img))
    buf_id.append(os.path.splitext(os.path.basename(path))[0])
    if len(buf_id) >= BATCH:
        flush()
    if i and i % 100 == 0:
        el = time.time() - t0
        print("  %4d/%d  %.1f img/s  eta %.0f min"
              % (i, len(dcm), i / el, (len(dcm) - i) / max(i / el, 1e-6) / 60), flush=True)
flush()

df = pd.DataFrame({"image_id": ids})
for k in HEADS:
    df["p_tb_%s" % k] = out[k]
df.to_csv(os.path.join(OUT, "zambia_scores.csv"), index=False)
print("\ndone in %.1f min | %d scored, %d skipped" % ((time.time() - t0) / 60, len(df), skipped))
print("wrote zambia_scores.csv:", list(df.columns))
df.head()
