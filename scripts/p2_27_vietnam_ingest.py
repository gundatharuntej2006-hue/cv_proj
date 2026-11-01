# -*- coding: utf-8 -*-
"""Ingest the Vietnam screening cohort: DICOM -> 512px -> embeddings, for three maskings.

2,708 participants from community active case-finding in Ho Chi Minh City, each with one
chest radiograph and an **Xpert MTB/RIF Ultra** result. 280 positive. CC0.

Why this cohort is the decisive test of the shortcut claim: the CSV labels the radiography
system, and **the scanner does not predict the outcome** (AUC 0.4722, chance - see
`docs/p2_vietnam_cohort_preflight.md`). In TBX11K, cases and controls were collected as
separate groups, so acquisition correlates with the label and a border-only model reaches
0.9094. Here it cannot: if border-only beats chance on this cohort it is not reading the
machine.

**DICOM handling, which is where this could silently go wrong.** The archives mix 12-bit
Konica and 14-bit DRTECH studies with different window centre/width. Scaling raw stored
values would manufacture exactly the scanner signature this dataset is valuable for not
having. So each image goes through the DICOM VOI LUT, is inverted if MONOCHROME1, is clipped
at the 0.5/99.5 percentiles, and only then is scaled to 8 bits and resized with the same
`INTER_AREA` call used for every other site. Some files leave `VOILUTFunction` empty, which
makes pydicom raise; those fall back to LINEAR.

We deliberately do **not** normalise harder to match TBX11K's intensity distribution. A
residual difference remains (per-scanner mean 119/125/164 against TBX11K's ~128) and that is
a real property of a new site, not something to tune away. Tuning preprocessing to the target
set would be a soft form of fitting to the test data.

**This is a test set. Nothing here is ever trained on.**

    py -3 scripts/p2_27_vietnam_ingest.py            # ingest + embed, resumable
    py -3 scripts/p2_27_vietnam_ingest.py --stage png  # standardised PNGs only
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p2_cls import embed as E                                       # noqa: E402

LIBS = Path("D:/dryad/libs")
CSV = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
PNG_DIR = ROOT / "data/processed/v1/vietnam_img512"
MASK_DIR = ROOT / "data/processed/v1/masks_vietnam"
OUT_DIR = ROOT / "artifacts/p2/external"
BORDER_FRAC = 0.12
VARIANTS = ("full", "lungs_only", "border_only")


def dicom_to_512(buf):
    """One DICOM byte-string -> uint8 512x512, via the VOI LUT. See the module note."""
    import cv2
    import pydicom
    from pydicom.pixels import apply_voi_lut
    d = pydicom.dcmread(io.BytesIO(buf))
    if d.get("VOILUTFunction", "") == "":
        d.VOILUTFunction = "LINEAR"          # blank is legal DICOM but pydicom raises
    try:
        a = apply_voi_lut(d.pixel_array, d).astype(np.float32)
    except Exception:
        a = d.pixel_array.astype(np.float32)
    if d.get("PhotometricInterpretation") == "MONOCHROME1":
        a = a.max() - a                       # MONOCHROME1 stores inverted
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    return cv2.resize(a.astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA)


def stage_png(log=print):
    """Decode every DICOM once and cache a standardised PNG. Resumable."""
    from PIL import Image
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    libs = sorted(LIBS.glob("*.zip"))
    if not libs:
        raise SystemExit("no DICOM archives under %s" % LIBS)
    done = {p.stem for p in PNG_DIR.glob("*.png")}
    n, t0, failed = 0, time.perf_counter(), []
    for lib in libs:
        with zipfile.ZipFile(lib) as z:
            for inf in z.infolist():
                if not inf.filename.endswith(".dcm"):
                    continue
                pid = inf.filename.split(".")[0]
                if pid in done:
                    continue
                try:
                    img = dicom_to_512(z.read(inf))
                except Exception as e:
                    failed.append((pid, "%s: %s" % (type(e).__name__, e)))
                    continue
                Image.fromarray(img).save(PNG_DIR / ("%s.png" % pid))
                n += 1
                if n % 250 == 0:
                    log("  %4d decoded  %.1f img/s"
                        % (n, n / max(1e-9, time.perf_counter() - t0)))
    log("decoded %d new images (%.1f min); failures: %d"
        % (n, (time.perf_counter() - t0) / 60, len(failed)))
    for pid, err in failed[:5]:
        log("    %s %s" % (pid, err))
    return failed


def stage_masks(log=print):
    """Lung masks via the same C05 segmenter used on TBX11K and the other sites."""
    import cv2
    from PIL import Image
    import p1_build_c04_c05 as C05
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(PNG_DIR.glob("*.png"))
            if not (MASK_DIR / p.name).exists()]
    if not todo:
        return 0
    import torch
    import torchxrayvision as xrv
    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval()
    log("segmenting %d images" % len(todo))
    t0, n_fail = time.perf_counter(), 0
    for k, p in enumerate(todo):
        a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        m = C05.lung_mask(model, a, torch, xrv)
        if m.shape != (512, 512):
            m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
        if not C05.roi_from_mask(m)["sanity_pass"]:
            n_fail += 1
        Image.fromarray((m * 255).astype(np.uint8)).save(MASK_DIR / p.name)
        if k % 400 == 0 and k:
            log("  %4d/%d" % (k, len(todo)))
    log("  segmented in %.1f min; sanity failures %d (%.1f%%)"
        % ((time.perf_counter() - t0) / 60, n_fail, 100 * n_fail / max(len(todo), 1)))
    # a second resident backbone is what exhausted memory last time
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n_fail


def mask_image(img, mask, variant):
    out = img.astype(np.float32).copy()
    fill = float(out.mean())
    if variant == "full":
        return img
    if variant == "border_only":
        h, w = out.shape
        bh, bw = int(round(h * BORDER_FRAC)), int(round(w * BORDER_FRAC))
        keep = np.zeros(out.shape, bool)
        keep[:bh, :] = keep[-bh:, :] = True
        keep[:, :bw] = keep[:, -bw:] = True
        out[~keep] = fill
    else:                                     # lungs_only
        if mask is None or mask.sum() == 0:
            return None
        out[~mask] = fill
    return np.clip(out, 0, 255).astype(np.uint8)


def stage_embed(variants, log=print):
    from PIL import Image
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pngs = sorted(PNG_DIR.glob("*.png"))
    todo = [v for v in variants
            if not (OUT_DIR / ("radino_vietnam_%s_v1.npz" % v)).exists()]
    if not todo:
        log("all variants already embedded")
        return
    embedder = E.Embedder(batch=16)           # one backbone, reused across variants
    for variant in todo:
        ids, vecs, buf, bids = [], [], [], []
        t0 = time.perf_counter()
        for p in pngs:
            img = np.array(Image.open(p).convert("L"), dtype=np.uint8)
            mask = None
            if variant == "lungs_only":
                mp = MASK_DIR / p.name
                if not mp.exists():
                    continue
                mask = np.array(Image.open(mp).convert("L"), np.uint8) > 127
            a = mask_image(img, mask, variant)
            if a is None:
                continue
            buf.append(a)
            bids.append(p.stem)
            if len(buf) >= 16:
                vecs.append(embedder.embed(buf).astype(np.float32))
                ids.extend(bids)
                buf, bids = [], []
        if buf:
            vecs.append(embedder.embed(buf).astype(np.float32))
            ids.extend(bids)
        arr = np.concatenate(vecs)
        out_p = OUT_DIR / ("radino_vietnam_%s_v1.npz" % variant)
        tmp = out_p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:           # savez would append .npz to the path
            np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U16"), emb=arr)
        os.replace(tmp, out_p)
        log("  %-12s embedded %d images in %.1f min"
            % (variant, len(ids), (time.perf_counter() - t0) / 60))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("png", "masks", "embed", "all"), default="all")
    args = ap.parse_args()

    meta = {"created": time.strftime("%Y-%m-%d")}
    if args.stage in ("png", "all"):
        failed = stage_png()
        meta["decode_failures"] = len(failed)
    if args.stage in ("masks", "all"):
        meta["segmentation_sanity_failures"] = stage_masks()
    if args.stage in ("embed", "all"):
        stage_embed(VARIANTS)

    n = len(list(PNG_DIR.glob("*.png")))
    meta["n_images"] = n
    print("\n%d standardised images at %s" % (n, PNG_DIR))
    json.dump(meta, open(OUT_DIR / "vietnam_ingest.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
