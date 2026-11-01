# -*- coding: utf-8 -*-
"""Kaggle notebook: score the frozen TBX11K model on the Zambia CIDRZ cohort.

Paste this into a Kaggle notebook. Nothing large is downloaded or uploaded: the 84 GB
dataset stays on Kaggle's servers, and what comes back is a CSV of a few hundred kilobytes.

SETUP IN THE KAGGLE UI
  1. New notebook -> Add Input -> search "Google + CIDRZ Health AI Evaluation Zambia"
     (googlehealthai/google-health-ai). It mounts read-only at /kaggle/input/.
  2. Settings -> Accelerator: GPU T4 x2 (or P100).
  3. Settings -> Internet: ON  (needed to fetch RAD-DINO and the segmenter weights).
  4. Add Input -> Upload -> `head_weights.npz` from artifacts/p2/bundle/head.npz,
     plus the lungs-only and border-only heads if running all three maskings.
  5. Run all. Download `zambia_scores.csv` from the output pane.

WHY THIS COHORT
  1,828 participants, ~192 bacteriologically confirmed TB (paired sputum culture and/or
  Xpert Ultra), consecutively enrolled at three Lusaka clinics, **35.5% HIV-positive**.
  Positives and negatives come from the same cohort and the same machines, so acquisition
  cannot encode the label.

  The HIV variable is the point. HIV-positive tuberculosis presents atypically - less
  cavitation, more lymphadenopathy, and sometimes a radiographically normal chest - and no
  other cohort we have carries it. Vietnam's depositors dropped it. A screening tool that
  silently fails in HIV-positive patients is an equity failure hiding inside an acceptable
  overall number.

  The dataset also carries 10 independent radiologist reads per image and Google's own
  TB/abnormality scores with prespecified thresholds, giving both a human-variation
  reference and a second commercial comparator alongside Lunit's 0.7827 on Vietnam.

NOTHING HERE IS TRAINED. The head comes from TBX11K and is applied unchanged.
"""
# ---------------------------------------------------------------- 0. environment
# !pip install -q pydicom torchxrayvision

import glob
import io
import os

import numpy as np
import pandas as pd
import torch

IN = "/kaggle/input"
OUT = "/kaggle/working"
BORDER_FRAC = 0.12
BATCH = 16

# ---------------------------------------------------------------- 1. locate the data
# The dataset layout is not documented ahead of time, so discover it rather than assume.
dcm_files = glob.glob(os.path.join(IN, "**", "*.dcm"), recursive=True)
csv_files = [p for p in glob.glob(os.path.join(IN, "**", "*.csv"), recursive=True)]
print("DICOM files found:", len(dcm_files))
print("CSV files found:")
for p in csv_files[:20]:
    print("   ", p)

# Inspect the metadata before relying on it: we need a participant key, the TB reference
# standard, and HIV status.
for p in csv_files[:5]:
    try:
        d = pd.read_csv(p, nrows=5)
        print("\n%s  ->  %s" % (os.path.basename(p), list(d.columns)[:25]))
    except Exception as e:
        print("could not read", p, e)

# ---------------------------------------------------------------- 2. DICOM -> 512px
def dicom_to_512(path_or_bytes):
    """Identical normalisation to the other four sites: VOI LUT, invert MONOCHROME1,
    percentile clip, INTER_AREA to 512. Do not change this without changing it everywhere."""
    import cv2
    import pydicom
    from pydicom.pixels import apply_voi_lut
    d = pydicom.dcmread(path_or_bytes)
    if d.get("VOILUTFunction", "") == "":
        d.VOILUTFunction = "LINEAR"          # blank is legal DICOM; pydicom raises on it
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


# ---------------------------------------------------------------- 3. the backbone
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "microsoft/rad-dino"
REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"   # pinned, as in D15
proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
backbone = AutoModel.from_pretrained(MODEL_ID, revision=REVISION)
dev = "cuda" if torch.cuda.is_available() else "cpu"
backbone = (backbone.half() if dev == "cuda" else backbone.float()).eval().to(dev)
for prm in backbone.parameters():
    prm.requires_grad_(False)


def embed(images):
    """list of uint8 (H,W) -> float32 [n, 768], L2-normalised. Matches embed.Embedder."""
    rgb = [np.stack([im] * 3, axis=-1) for im in images]
    x = proc(images=rgb, return_tensors="pt")["pixel_values"]
    x = x.to(dev, dtype=torch.float16 if dev == "cuda" else torch.float32)
    with torch.inference_mode():
        out = backbone(pixel_values=x)
        f = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]
        f = f.float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return f.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- 4. the frozen head
def load_head(path):
    with np.load(path) as z:
        return z["W"].astype(np.float32), z["b"].astype(np.float32)


def p_tb(emb, W, b):
    logits = emb @ W.T + b
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


HEADS = {}
for name, fn in (("full", "head.npz"),
                 ("lungs_only", "head_lungs_only.npz"),
                 ("border_only", "head_border_only.npz")):
    hits = glob.glob(os.path.join(IN, "**", fn), recursive=True)
    if hits:
        HEADS[name] = load_head(hits[0])
        print("loaded head:", name, "from", hits[0])
if not HEADS:
    raise SystemExit("no head weights found - upload head.npz as a Kaggle dataset input")

# ---------------------------------------------------------------- 5. lung masks (optional)
SEGMENT = "lungs_only" in HEADS
if SEGMENT:
    import torchxrayvision as xrv
    seg = xrv.baseline_models.chestx_det.PSPNet().eval()

    def lung_mask(img_u8):
        x = xrv.datasets.normalize(img_u8.astype(np.float32), 255)
        t = torch.from_numpy(x)[None, None, ...]
        with torch.no_grad():
            o = seg(t)[0]
        tgt = {n: i for i, n in enumerate(seg.targets)}
        pr = torch.sigmoid(o)
        return ((pr[tgt["Left Lung"]] > 0.5) | (pr[tgt["Right Lung"]] > 0.5)).cpu().numpy()

# ---------------------------------------------------------------- 6. score every image
rows, buf, bufid = [], {k: [] for k in HEADS}, []
results = {k: [] for k in HEADS}
ids = []

for i, path in enumerate(sorted(dcm_files)):
    try:
        img = dicom_to_512(path)
    except Exception as e:
        print("skip", os.path.basename(path), type(e).__name__)
        continue
    variants = {"full": img}
    if "border_only" in HEADS:
        variants["border_only"] = mask_border(img)
    if SEGMENT:
        m = lung_mask(img)
        if m.sum():
            o = img.astype(np.float32).copy()
            o[~m] = float(o.mean())
            variants["lungs_only"] = np.clip(o, 0, 255).astype(np.uint8)
        else:
            variants["lungs_only"] = img
    for k in HEADS:
        buf[k].append(variants.get(k, img))
    bufid.append(os.path.splitext(os.path.basename(path))[0])

    if len(bufid) >= BATCH:
        for k, (W, b) in HEADS.items():
            results[k].extend(p_tb(embed(buf[k]), W, b).tolist())
            buf[k] = []
        ids.extend(bufid)
        bufid = []
    if i % 200 == 0:
        print("  %d/%d" % (i, len(dcm_files)), flush=True)

if bufid:
    for k, (W, b) in HEADS.items():
        results[k].extend(p_tb(embed(buf[k]), W, b).tolist())
    ids.extend(bufid)

out = pd.DataFrame({"image_id": ids})
for k in HEADS:
    out["p_tb_%s" % k] = results[k]
out.to_csv(os.path.join(OUT, "zambia_scores.csv"), index=False)
print("\nwrote zambia_scores.csv with %d rows and columns %s"
      % (len(out), list(out.columns)))

# ---------------------------------------------------------------- 7. join and report
# Join to whichever metadata CSV carries the TB reference standard and HIV status, then
# download zambia_scores.csv. The AUCs, the HIV stratification and the comparison against
# Google's own scores are computed locally, where the other four sites already live.
print("\nNext: download zambia_scores.csv and the metadata CSV, then run the local analysis.")
