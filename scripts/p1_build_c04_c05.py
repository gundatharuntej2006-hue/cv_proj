# -*- coding: utf-8 -*-
"""Build C04 (preprocessed image store) and C05 (lung ROI table) for TBX11K.

These are Person 1's contracts. This builds them locally so the rest of the pipeline can
run while C04/C05 are outstanding — it is an unblock and a reference implementation, not a
claim on P1's ownership. When P1 ships the real C04/C05, point the downstream scripts at
those instead; the schemas here follow the contract so the swap is clean.

C04 — `data/processed/v1/tbx11k/img512/{image_id}.png` and `.../img512_clahe/{image_id}.png`
      plus `manifest.parquet`. TBX11K already ships 512x512 PNGs, so img512 is a verified
      copy; the CLAHE variant uses the parameters fixed a priori (clip 2.0, tile 8x8).

C05 — `data/processed/v1/lung_roi.parquet` (+ masks). Lung fields come from the pretrained
      TorchXRayVision ChestX-Det PSPNet (Apache-2.0), which is what the methodology
      specifies: a pretrained segmenter plus a sanity check, never a model we train.

Sanity check (fails -> crop falls back to the full image, recorded, never silently):
  area      lung area outside 10%-70% of the frame
  components fewer than 2 connected lung regions
  bbox_size  lung box narrower/shorter than 30% of the frame
  truncated  lung box touching the frame edge on 3+ sides

    py -3 scripts/p1_build_c04_c05.py --limit 200     # smoke
    py -3 scripts/p1_build_c04_c05.py                 # all trainval
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/tbx11k/TBX11K/imgs"
PROC = ROOT / "data/processed/v1"
MARGINS = (0.00, 0.05, 0.10, 0.15)
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)


def clahe(img_u8):
    import cv2
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(img_u8)


def lung_mask(model, img_u8, torch, xrv):
    """Binary lung mask (left|right) at 512x512 from the pretrained segmenter."""
    x = xrv.datasets.normalize(img_u8.astype(np.float32), 255)   # -> [-1024, 1024]
    t = torch.from_numpy(x)[None, None, ...]
    with torch.no_grad():
        out = model(t)[0]                                        # [14, H, W] logits
    tgt = {n: i for i, n in enumerate(model.targets)}
    li, ri = tgt.get("Left Lung"), tgt.get("Right Lung")
    prob = torch.sigmoid(out)
    m = ((prob[li] > 0.5) | (prob[ri] > 0.5)).cpu().numpy().astype(np.uint8)
    return m


def roi_from_mask(m):
    """Lung bbox, area fraction, connected-component count, and sanity verdict."""
    from scipy import ndimage
    H, W = m.shape
    area_frac = float(m.mean())
    lab, n_comp = ndimage.label(m)
    # keep components >=1% of frame, to ignore speckle
    sizes = ndimage.sum(m, lab, range(1, n_comp + 1))
    big = [i + 1 for i, s in enumerate(sizes) if s >= 0.01 * H * W]
    n_big = len(big)
    if not big:
        return dict(area_frac=area_frac, n_components=0, box=(0, 0, W, H),
                    sanity_pass=False, reasons="area")
    keep = np.isin(lab, big)
    ys, xs = np.where(keep)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    reasons = []
    if not (0.10 <= area_frac <= 0.70):
        reasons.append("area")
    if n_big < 2:
        reasons.append("components")
    if (x1 - x0) < 0.30 * W or (y1 - y0) < 0.30 * H:
        reasons.append("bbox_size")
    edges = (x0 <= 1) + (y0 <= 1) + (x1 >= W - 1) + (y1 >= H - 1)
    if edges >= 3:
        reasons.append("truncated")
    return dict(area_frac=area_frac, n_components=n_big, box=(x0, y0, x1, y1),
                sanity_pass=not reasons, reasons="|".join(reasons))


def crop_box(box, margin, W=512, H=512):
    x0, y0, x1, y1 = box
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(int(round(x0 - mw)), 0), max(int(round(y0 - mh)), 0),
            min(int(round(x1 + mw)), W), min(int(round(y1 + mh)), H))


def main():
    ap = argparse.ArgumentParser(description="Build C04 + C05 for TBX11K")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labels", default="C:/Users/atuld/Downloads/P2_handoff_for_C01/labels_tbx11k_trainval_v1.csv")
    a = ap.parse_args()

    import cv2
    import pandas as pd
    import torch
    import torchxrayvision as xrv
    from PIL import Image

    lab = pd.read_csv(a.labels)
    if a.limit:
        lab = lab.head(a.limit)
    (PROC / "tbx11k/img512").mkdir(parents=True, exist_ok=True)
    (PROC / "tbx11k/img512_clahe").mkdir(parents=True, exist_ok=True)
    (PROC / "masks").mkdir(parents=True, exist_ok=True)

    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    print("segmenter on %s | %d images" % (dev, len(lab)))

    man, roi = [], []
    import time
    # resumable: a long segmentation pass will be interrupted, and restarting from zero
    # wastes the GPU time already spent. Images whose outputs all exist are skipped.
    done = {p.stem for p in (PROC / "masks").glob("*.png")}
    if done:
        print("resuming: %d images already segmented" % len(done))
    t0 = time.perf_counter()
    for k, r in enumerate(lab.itertuples(), 1):
        if r.image_id in done:
            # segmentation already paid for: rebuild the rows from the saved mask, no GPU
            mp_ = PROC / "masks" / ("%s.png" % r.image_id)
            m_ = (np.array(Image.open(mp_).convert("L")) > 127).astype(np.uint8)
            info_ = roi_from_mask(m_)
            man.append({"image_id": r.image_id, "dataset": "tbx11k",
                        "path_img512": "data/processed/v1/tbx11k/img512/%s.png" % r.image_id,
                        "path_img512_clahe": "data/processed/v1/tbx11k/img512_clahe/%s.png" % r.image_id,
                        "orig_h": 512, "orig_w": 512, "src_format": "png",
                        "decode_status": "ok", "error_msg": "", "phi_tags_removed": 0})
            for mg_ in MARGINS:
                cb_ = crop_box(info_["box"], mg_) if info_["sanity_pass"] else (0, 0, 512, 512)
                roi.append({"image_id": r.image_id,
                            "mask_path": "data/processed/v1/masks/%s.png" % r.image_id,
                            "lung_area_frac": round(info_["area_frac"], 4),
                            "n_components": info_["n_components"],
                            "lung_x0": info_["box"][0], "lung_y0": info_["box"][1],
                            "lung_x1": info_["box"][2], "lung_y1": info_["box"][3],
                            "sanity_pass": info_["sanity_pass"], "fail_reasons": info_["reasons"],
                            "margin": mg_,
                            "crop_x0": cb_[0], "crop_y0": cb_[1], "crop_x1": cb_[2], "crop_y1": cb_[3],
                            "fallback_full": not info_["sanity_pass"]})
            continue
        src = ROOT / "data/raw/tbx11k/TBX11K" / r.rel_path
        img = np.array(Image.open(src).convert("L"), dtype=np.uint8)
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        p512 = PROC / "tbx11k/img512" / ("%s.png" % r.image_id)
        pcl = PROC / "tbx11k/img512_clahe" / ("%s.png" % r.image_id)
        Image.fromarray(img).save(p512)
        Image.fromarray(clahe(img)).save(pcl)

        x = xrv.datasets.normalize(img.astype(np.float32), 255)
        t = torch.from_numpy(x)[None, None, ...].to(dev)
        with torch.no_grad():
            out = model(t)[0]
        tgt = {n: i for i, n in enumerate(model.targets)}
        prob = torch.sigmoid(out)
        m = ((prob[tgt["Left Lung"]] > 0.5) | (prob[tgt["Right Lung"]] > 0.5)).cpu().numpy().astype(np.uint8)
        mp = PROC / "masks" / ("%s.png" % r.image_id)
        Image.fromarray((m * 255).astype(np.uint8)).save(mp)

        info = roi_from_mask(m)
        man.append({"image_id": r.image_id, "dataset": "tbx11k",
                    "path_img512": str(p512.relative_to(ROOT)).replace("\\", "/"),
                    "path_img512_clahe": str(pcl.relative_to(ROOT)).replace("\\", "/"),
                    "orig_h": 512, "orig_w": 512, "src_format": "png",
                    "decode_status": "ok", "error_msg": "", "phi_tags_removed": 0})
        for mg in MARGINS:
            cb = crop_box(info["box"], mg) if info["sanity_pass"] else (0, 0, 512, 512)
            roi.append({"image_id": r.image_id,
                        "mask_path": str(mp.relative_to(ROOT)).replace("\\", "/"),
                        "lung_area_frac": round(info["area_frac"], 4),
                        "n_components": info["n_components"],
                        "lung_x0": info["box"][0], "lung_y0": info["box"][1],
                        "lung_x1": info["box"][2], "lung_y1": info["box"][3],
                        "sanity_pass": info["sanity_pass"], "fail_reasons": info["reasons"],
                        "margin": mg,
                        "crop_x0": cb[0], "crop_y0": cb[1], "crop_x1": cb[2], "crop_y1": cb[3],
                        "fallback_full": not info["sanity_pass"]})
        if k % 250 == 0:
            el = time.perf_counter() - t0
            print("  %5d/%d  %.1f img/s  eta %.1f min" % (k, len(lab), k / el, (len(lab) - k) / (k / el) / 60))

    # merge with any rows from previous runs so the tables are complete after a resume
    cache = PROC / "_rows_cache.parquet"
    rcache = PROC / "_roi_cache.parquet"
    if cache.exists() and man:
        man = pd.concat([pd.read_parquet(cache), pd.DataFrame(man)]).drop_duplicates("image_id").to_dict("records")
        roi = pd.concat([pd.read_parquet(rcache), pd.DataFrame(roi)]).drop_duplicates(["image_id", "margin"]).to_dict("records")
    elif cache.exists():
        man = pd.read_parquet(cache).to_dict("records")
        roi = pd.read_parquet(rcache).to_dict("records")
    pd.DataFrame(man).to_parquet(cache)
    pd.DataFrame(roi).to_parquet(rcache)
    pd.DataFrame(man).to_parquet(PROC / "manifest.parquet")
    rdf = pd.DataFrame(roi)
    rdf.to_parquet(PROC / "lung_roi.parquet")
    sp = rdf[rdf.margin == 0.10]
    print("\nC04 manifest : %s (%d rows)" % (PROC / "manifest.parquet", len(man)))
    print("C05 lung_roi : %s (%d rows, %d images x %d margins)"
          % (PROC / "lung_roi.parquet", len(rdf), len(man), len(MARGINS)))
    print("sanity pass rate: %.1f%%  (fallback to full image otherwise)"
          % (100 * sp.sanity_pass.mean()))
    if (~sp.sanity_pass).any():
        from collections import Counter
        print("fail reasons:", dict(Counter(r for s in sp[~sp.sanity_pass].fail_reasons for r in s.split("|") if r)))
    print("median lung area frac: %.3f" % sp.lung_area_frac.median())


if __name__ == "__main__":
    sys.exit(main())
