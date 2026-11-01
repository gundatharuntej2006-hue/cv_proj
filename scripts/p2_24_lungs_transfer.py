# -*- coding: utf-8 -*-
"""The positive control: does a LUNGS-ONLY model transfer to Shenzhen?

`p2_shortcut_does_not_transfer.md` established the negative half — the border shortcut is at
chance at Shenzhen (0.5209), so the full model's external 0.9382 cannot be explained by it.
That shows what is *not* carrying the external result. It does not yet show what is.

This is the other half. Segment the 800 external radiographs with the **same** ChestX-Det
PSPNet used to build C05, mask everything outside the lung field exactly as
`p2_19_shortcut_probe.py` does, and apply the lungs-only head trained on TBX11K unchanged.

  - lungs-only transfers (AUC near the full model's 0.9382) -> the external result is
    carried by lung tissue, and the argument closes: shortcut dead, lungs alive.
  - lungs-only also collapses -> the external result comes from something that is neither
    the border nor the lung field, and we would not yet know what it is.

The segmentation sanity check from C05 (`roi_from_mask`) is applied unchanged, and images
that fail it are reported rather than silently dropped: a segmenter trained on one
population can fail on another, and that failure rate is itself worth knowing before anyone
relies on the lung-crop pipeline at a new site.

    py -3 scripts/p2_24_lungs_transfer.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p2_cls import embed as E                                       # noqa: E402
from p2_cls import heads as H                                       # noqa: E402
from p2_cls import registry as R                                    # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

FLAT = ROOT / "data/raw/_external_flat"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT_DIR = ROOT / "artifacts/p2/shortcut"
MASK_DIR = ROOT / "data/processed/v1/masks_external"
EMB_OUT = OUT_DIR / "radino_nlm_lungs_only_v1.npz"
TBX_LUNGS = OUT_DIR / "radino_tbx11k_lungs_only_v1.npz"
TBX_FULL = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
OUT = OUT_DIR / "lungs_transfer.json"
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 2000


def segment_external(log=print):
    """Lung masks for the 800 NLM images, via the C05 segmenter. Resumable."""
    import cv2
    from PIL import Image
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in FLAT.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    todo = [p for p in paths if not (MASK_DIR / (p.stem + ".png")).exists()]
    if todo:
        import torch
        import torchxrayvision as xrv
        import p1_build_c04_c05 as C05
        model = xrv.baseline_models.chestx_det.PSPNet()
        model.eval()
        log("segmenting %d external images" % len(todo))
        t0 = time.perf_counter()
        for k, p in enumerate(todo):
            a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
            if a.shape != (512, 512):
                a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
            m = C05.lung_mask(model, a, torch, xrv)
            if m.shape != (512, 512):
                m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
            Image.fromarray((m * 255).astype(np.uint8)).save(MASK_DIR / (p.stem + ".png"))
            if k % 100 == 0 and k:
                log("  %4d/%d  %.1f img/s" % (k, len(todo),
                                              k / max(1e-9, time.perf_counter() - t0)))
        log("  done in %.1f min" % ((time.perf_counter() - t0) / 60))
    return paths


def build_and_embed(paths, log=print):
    """Lungs-only versions of the external images, embedded. Returns ids, emb, sanity info."""
    import cv2
    from PIL import Image
    import p1_build_c04_c05 as C05
    if EMB_OUT.exists():
        with np.load(EMB_OUT) as z:
            info = json.load(open(OUT_DIR / "nlm_lungs_only_sanity.json", encoding="utf-8"))
            return z["image_id"].tolist(), z["emb"].astype(np.float32), info

    emb = E.Embedder(batch=16)
    ids, vecs, buf, bids = [], [], [], []
    failed, areas = [], []

    def flush():
        if buf:
            vecs.append(emb.embed(buf).astype(np.float32))
            ids.extend(bids)
            buf.clear()
            bids.clear()

    t0 = time.perf_counter()
    for p in paths:
        a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        if a.shape != (512, 512):
            a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
        m = np.array(Image.open(MASK_DIR / (p.stem + ".png")).convert("L"), np.uint8) > 127
        roi = C05.roi_from_mask(m.astype(np.uint8))
        areas.append(roi["area_frac"])
        if not roi["sanity_pass"]:
            failed.append({"image_id": p.stem, "reasons": roi["reasons"],
                           "area_frac": round(roi["area_frac"], 4)})
        if m.sum() == 0:
            continue                      # nothing to keep; cannot build a lungs-only image
        out = a.astype(np.float32).copy()
        out[~m] = float(out.mean())       # identical rule to p2_19_shortcut_probe
        buf.append(np.clip(out, 0, 255).astype(np.uint8))
        bids.append(p.stem)
        if len(buf) >= 16:
            flush()
    flush()
    arr = np.concatenate(vecs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EMB_OUT.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U48"), emb=arr)
    import os
    os.replace(tmp, EMB_OUT)
    info = {"n": len(paths), "n_embedded": len(ids), "n_sanity_failed": len(failed),
            "sanity_fail_rate": round(len(failed) / max(len(paths), 1), 4),
            "median_lung_area_frac": round(float(np.median(areas)), 4),
            "examples": failed[:10]}
    json.dump(info, open(OUT_DIR / "nlm_lungs_only_sanity.json", "w", encoding="utf-8"),
              indent=1)
    log("  embedded %d/%d (%.1f min); segmentation sanity failures: %d (%.1f%%)"
        % (len(ids), len(paths), (time.perf_counter() - t0) / 60,
           len(failed), 100 * len(failed) / max(len(paths), 1)))
    return ids, arr, info


def train_head(path, log=print):
    import pandas as pd
    with np.load(path) as z:
        ids, X = z["image_id"].tolist(), z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    tr, dv = c01[c01.split == "train"], c01[c01.split == "development"]
    Xtr, ytr = X[[idx[i] for i in tr.image_id]], H.binary_labels(tr.category.tolist())
    Xdv, ydv = X[[idx[i] for i in dv.image_id]], H.binary_labels(dv.category.tolist())
    res, auc, sd = H.train_seeds(
        Xtr, ytr, Xdv, ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    return (res[0].state_dict["weight"].numpy().astype(np.float32),
            res[0].state_dict["bias"].numpy().astype(np.float32), float(auc))


def boot_auc(y, p, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    vals = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        vals.append(SEL.auc(y[k], p[k]))
    return (float(SEL.auc(y, p)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def parse_id(s):
    site, _, rest = s.partition("__")
    return site, int(rest.rsplit("_", 1)[1])


def main():
    paths = segment_external()
    ids, X, info = build_and_embed(paths)
    print("\ntraining the lungs-only head on TBX11K:")
    W, b, dev_auc = train_head(TBX_LUNGS)
    print("  lungs_only TBX11K development AUC %.4f" % dev_auc)

    sites = np.array([parse_id(i)[0] for i in ids])
    y = np.array([parse_id(i)[1] for i in ids])
    p = R.p_tb_from_logits(R.head_logits(X, W, b), 2)

    rows = []
    print("\nlungs-only model applied to the external sites:")
    for key, name in (("shz", "Shenzhen"), ("mcu", "Montgomery")):
        m = sites == key
        a, lo, hi = boot_auc(y[m], p[m])
        rows.append({"site": name, "n": int(m.sum()), "n_tb": int(y[m].sum()),
                     "lungs_only_auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)]})
        print("  %-12s n=%3d  AUC %.4f [%.4f, %.4f]" % (name, m.sum(), a, lo, hi))

    # the three-way comparison that closes the argument
    prev = json.load(open(OUT_DIR / "shortcut_transfer.json", encoding="utf-8"))
    prev_by = {(r["trained_and_tested_on"], r["site"]): r for r in prev["external"]}
    print("\n  %-12s %10s %12s %12s" % ("site", "full", "lungs-only", "border-only"))
    for r in rows:
        f = prev_by[("full", r["site"])]["auc"]
        bo = prev_by[("border_only", r["site"])]["auc"]
        r["full_auc"], r["border_only_auc"] = f, bo
        print("  %-12s %10.4f %12.4f %12.4f" % (r["site"], f, r["lungs_only_auc"], bo))

    shz = next(r for r in rows if r["site"] == "Shenzhen")
    print("\ninterpretation (Shenzhen, the larger set):")
    if shz["lungs_only_auc"] >= 0.85 and shz["border_only_auc"] < 0.65:
        verdict = ("CLOSED: lung tissue carries the external result and the border does "
                   "not. The external AUC reflects pathology.")
    elif shz["lungs_only_auc"] >= 0.75:
        verdict = ("MOSTLY: lungs carry much of the external result, but less than the "
                   "full image, so some signal lies outside both regions tested.")
    else:
        verdict = ("OPEN: neither the lung field nor the border explains the external "
                   "result on its own.")
    print("  lungs-only %.4f vs border-only %.4f vs full %.4f"
          % (shz["lungs_only_auc"], shz["border_only_auc"], shz["full_auc"]))
    print("  -> %s" % verdict)
    if info["sanity_fail_rate"] > 0.05:
        print("  NOTE: the C05 segmentation sanity check failed on %.1f%% of external "
              "images,\n  so the lung masks are less reliable here than on TBX11K."
              % (100 * info["sanity_fail_rate"]))

    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "tbx11k_lungs_only_dev_auc": round(dev_auc, 4),
           "segmentation": info, "external": rows, "verdict": verdict,
           "note": ("EXPLORATORY. Same PSPNet segmenter and masking rule as C05 and the "
                    "shortcut probe; head trained on TBX11K and applied unchanged. "
                    "internal_test untouched.")}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
