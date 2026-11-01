# -*- coding: utf-8 -*-
"""DA and DB (NITRD, New Delhi) as a third external site.

Two sites let you say "transfer degrades with distance"; three let you say it with a
straight face. DA and DB ship inside the TBX11K archive under imgs/extra/da+db/, are
referenced only by the all_*.txt lists, and are therefore **absent from our core-list
training set** - the same situation as Shenzhen and Montgomery.

They are already 512x512 because TBX11K standardised them on the way in, which removes the
preprocessing mismatch that contaminated two earlier cross-site comparisons here.

Labels are in the filenames: n*/nx* normal, p*/px* positive. DA is the .png pair, DB the
.jpg pair, and they are reported separately as well as pooled because they are two
different collections.

Runs the same three maskings as the other sites, so the numbers drop straight into the
comparison table: full image, lungs only (segmented here with the C05 segmenter), and
border only.

    py -3 scripts/p2_26_dadb_third_site.py
"""
import json
import os
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

SRC = ROOT / "data/raw/tbx11k/TBX11K/imgs/extra/da+db"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT_DIR = ROOT / "artifacts/p2/external"
SHORT = ROOT / "artifacts/p2/shortcut"
MASK_DIR = ROOT / "data/processed/v1/masks_dadb"
OUT = OUT_DIR / "dadb_third_site.json"
TBX = {"full": ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz",
       "lungs_only": SHORT / "radino_tbx11k_lungs_only_v1.npz",
       "border_only": SHORT / "radino_tbx11k_border_only_v1.npz"}
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 2000
BORDER_FRAC = 0.12


def images():
    """(path, site, label) for every DA/DB image. Label is in the filename."""
    out = []
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        n = p.name
        site = "DA" if p.suffix.lower() == ".png" else "DB"
        label = 1 if n[0] == "p" else 0
        out.append((p, site, label))
    return out


def mask_border(img):
    out = img.astype(np.float32).copy()
    h, w = out.shape
    bh, bw = int(round(h * BORDER_FRAC)), int(round(w * BORDER_FRAC))
    keep = np.zeros(out.shape, bool)
    keep[:bh, :] = keep[-bh:, :] = True
    keep[:, :bw] = keep[:, -bw:] = True
    out[~keep] = float(out.mean())
    return np.clip(out, 0, 255).astype(np.uint8)


def segment(paths, log=print):
    import cv2
    from PIL import Image
    import p1_build_c04_c05 as C05
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p for p, _s, _y in paths if not (MASK_DIR / (p.stem + ".png")).exists()]
    if not todo:
        return
    import torch
    import torchxrayvision as xrv
    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval()
    log("segmenting %d DA/DB images" % len(todo))
    t0 = time.perf_counter()
    for p in todo:
        a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        if a.shape != (512, 512):
            a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
        m = C05.lung_mask(model, a, torch, xrv)
        if m.shape != (512, 512):
            m = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
        Image.fromarray((m * 255).astype(np.uint8)).save(MASK_DIR / (p.stem + ".png"))
    log("  done in %.1f min" % ((time.perf_counter() - t0) / 60))
    # release the segmenter before a second backbone is loaded; on a 6 GB laptop two
    # resident models is the difference between running and an allocation failure
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_variant(paths, variant, emb, log=print):
    import cv2
    from PIL import Image
    import p1_build_c04_c05 as C05
    out_p = OUT_DIR / ("radino_dadb_%s_v1.npz" % variant)
    if out_p.exists():
        with np.load(out_p) as z:
            return z["emb"].astype(np.float32), int(z["n_sanity_fail"])
    vecs, buf, n_fail = [], [], 0
    for p, _site, _y in paths:
        a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        if a.shape != (512, 512):
            a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
        if variant == "border_only":
            a = mask_border(a)
        elif variant == "lungs_only":
            m = np.array(Image.open(MASK_DIR / (p.stem + ".png")).convert("L"),
                         np.uint8) > 127
            if not C05.roi_from_mask(m.astype(np.uint8))["sanity_pass"]:
                n_fail += 1
            if m.sum():
                o = a.astype(np.float32).copy()
                o[~m] = float(o.mean())
                a = np.clip(o, 0, 255).astype(np.uint8)
        buf.append(a)
        if len(buf) >= 16:
            vecs.append(emb.embed(buf).astype(np.float32))
            buf = []
    if buf:
        vecs.append(emb.embed(buf).astype(np.float32))
    arr = np.concatenate(vecs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as fh:
        np.savez_compressed(fh, emb=arr, n_sanity_fail=np.array(n_fail))
    log("  embedded %d images (%s), segmentation sanity failures: %d"
        % (len(arr), variant, n_fail))
    return arr, n_fail


def train_head(path):
    import pandas as pd
    with np.load(path) as z:
        ids, X = z["image_id"].tolist(), z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    tr, dv = c01[c01.split == "train"], c01[c01.split == "development"]
    res, auc, _sd = H.train_seeds(
        X[[idx[i] for i in tr.image_id]], H.binary_labels(tr.category.tolist()),
        X[[idx[i] for i in dv.image_id]], H.binary_labels(dv.category.tolist()),
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


def main():
    paths = images()
    sites = np.array([s for _p, s, _y in paths])
    y = np.array([lab for _p, _s, lab in paths])
    print("DA/DB: %d images, %d TB-positive (DA %d, DB %d)"
          % (len(paths), y.sum(), (sites == "DA").sum(), (sites == "DB").sum()))

    segment(paths)
    embedder = E.Embedder(batch=16)        # loaded once and reused across all variants
    rows = []
    for variant in ("full", "lungs_only", "border_only"):
        if not TBX[variant].exists():
            print("skipping %s (TBX11K embeddings missing)" % variant)
            continue
        X, n_fail = embed_variant(paths, variant, embedder)
        W, b, dev_auc = train_head(TBX[variant])
        p = R.p_tb_from_logits(R.head_logits(X, W, b), 2)
        for grp in ("DA+DB", "DA", "DB"):
            m = np.ones(len(y), bool) if grp == "DA+DB" else (sites == grp)
            if len(set(y[m].tolist())) < 2:
                continue
            a, lo, hi = boot_auc(y[m], p[m])
            rows.append({"variant": variant, "site": grp, "n": int(m.sum()),
                         "n_tb": int(y[m].sum()), "auc": round(a, 4),
                         "ci95": [round(lo, 4), round(hi, 4)],
                         "tbx11k_dev_auc": round(dev_auc, 4),
                         "segmentation_sanity_failures": n_fail})
            print("  %-12s %-6s n=%3d TB=%2d  AUC %.4f [%.4f, %.4f]"
                  % (variant, grp, m.sum(), y[m].sum(), a, lo, hi))

    def get(v, s="DA+DB"):
        for r in rows:
            if r["variant"] == v and r["site"] == s:
                return r["auc"]
        return None

    print("\nthe three-site picture (pooled DA+DB):")
    print("  %-14s %10s %12s %12s" % ("site", "full", "lungs-only", "border-only"))
    print("  %-14s %10.4f %12.4f %12.4f" % ("DA+DB (India)", get("full") or 0,
                                            get("lungs_only") or 0, get("border_only") or 0))
    print("  (Shenzhen 0.9382 / 0.9284 / 0.5209 ; Montgomery 0.8183 / 0.8554 / 0.6082)")
    bo = get("border_only")
    if bo is not None:
        print("\n  border-only at a third site: %.4f -> %s"
              % (bo, "shortcut still dead" if bo < 0.65 else
                 "shortcut partially present here"))

    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "n_images": len(paths), "n_tb": int(y.sum()), "results": rows,
           "note": ("DA/DB ship inside the TBX11K archive under imgs/extra/, referenced "
                    "only by all_*.txt, so they are absent from our core-list training "
                    "set. Already 512x512 as distributed. EXPLORATORY; the reportable "
                    "external number must be a sealed run.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
