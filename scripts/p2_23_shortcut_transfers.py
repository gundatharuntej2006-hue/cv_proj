# -*- coding: utf-8 -*-
"""Does the border shortcut transfer to Shenzhen? This decides what the external AUC means.

Two established facts that interact badly:

  1. The classifier reads the border. Trained and tested on TBX11K, the outer 12% frame
     alone gives AUC 0.9094 (`p2_shortcut_probe.md`).
  2. External validation on Shenzhen gives AUC 0.9382 (`p2_external_validation.md`), which
     was read as evidence that the system generalises to a new hospital.

But Shenzhen, like TBX11K, collected its TB cases and its normal controls as separate
groups. If those two groups differ in *how* their images were acquired, the border trick
could work at Shenzhen too, and the 0.9382 would be the same shortcut succeeding twice
rather than the model recognising disease.

The test is a dissociation. Train a head on TBX11K **border-only** images, then score
Shenzhen **border-only** images with it:

  - border-only transfers (AUC near 0.93)  -> the external number is largely the shortcut;
  - border-only collapses (AUC near 0.5)   -> the shortcut does NOT transfer, so the full
    model's 0.9382 on Shenzhen must come from something else, i.e. real lung signal. That
    is far stronger evidence for the system than the 0.9382 alone.

Both the full-image and lungs-only heads are scored on the same external images, so the
three numbers can be read side by side.

    py -3 scripts/p2_23_shortcut_transfers.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_cls import embed as E                                       # noqa: E402
from p2_cls import heads as H                                       # noqa: E402
from p2_cls import registry as R                                    # noqa: E402
from p2_cls import selection as SEL                                 # noqa: E402

FLAT = ROOT / "data/raw/_external_flat"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT_DIR = ROOT / "artifacts/p2/shortcut"
OUT = OUT_DIR / "shortcut_transfer.json"
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 2000
BORDER_FRAC = 0.12          # must match p2_19_shortcut_probe.py

# TBX11K embeddings per masking variant (training side)
TBX = {
    "full":        ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz",
    "lungs_only":  OUT_DIR / "radino_tbx11k_lungs_only_v1.npz",
    "border_only": OUT_DIR / "radino_tbx11k_border_only_v1.npz",
}


def mask_border(img):
    """Keep only the outer BORDER_FRAC frame; fill the centre with the image mean.

    No lung mask is involved: the kept region is the same fixed rectangle for every image,
    which is what makes this variant immune to segmentation-quality objections.
    """
    out = img.astype(np.float32).copy()
    h, w = out.shape
    bh, bw = int(round(h * BORDER_FRAC)), int(round(w * BORDER_FRAC))
    keep = np.zeros(out.shape, bool)
    keep[:bh, :] = keep[-bh:, :] = True
    keep[:, :bw] = keep[:, -bw:] = True
    out[~keep] = float(out.mean())
    return np.clip(out, 0, 255).astype(np.uint8)


def embed_external(variant, log=print):
    """The 800 NLM images, standardised like C04, optionally border-masked, then embedded."""
    out_p = OUT_DIR / ("radino_nlm_%s_v1.npz" % variant)
    if out_p.exists():
        with np.load(out_p) as z:
            return z["image_id"].tolist(), z["emb"].astype(np.float32)
    import cv2
    from PIL import Image
    paths = sorted(p for p in FLAT.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    emb = E.Embedder(batch=16)
    ids, vecs, buf, bids = [], [], [], []

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
        buf.append(mask_border(a) if variant == "border_only" else a)
        bids.append(p.stem)
        if len(buf) >= 16:
            flush()
    flush()
    arr = np.concatenate(vecs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(".tmp")
    with open(tmp, "wb") as fh:                 # savez would append .npz to the path
        np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U48"), emb=arr)
    import os
    os.replace(tmp, out_p)
    log("  embedded %d external images (%s) in %.1f min"
        % (len(ids), variant, (time.perf_counter() - t0) / 60))
    return ids, arr


def train_head(variant, log=print):
    """A head trained on TBX11K under one masking variant; returns (W, b, dev AUC)."""
    import pandas as pd
    p = TBX[variant]
    if not p.exists():
        raise SystemExit("missing %s; run scripts/p2_19_shortcut_probe.py --stage embed" % p)
    with np.load(p) as z:
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
    W = res[0].state_dict["weight"].numpy().astype(np.float32)
    b = res[0].state_dict["bias"].numpy().astype(np.float32)
    log("  %-12s TBX11K development AUC %.4f (seed sd %.4f)" % (variant, auc, sd))
    return W, b, float(auc)


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
    print("training heads on TBX11K:")
    heads = {v: train_head(v) for v in ("full", "lungs_only", "border_only")}

    print("\nembedding the external sets:")
    ext = {"full": embed_external("full"), "border_only": embed_external("border_only")}

    rows = []
    print("\nexternal AUC by what the model was given (bootstrap %d):" % N_BOOT)
    print("  %-14s %-12s %9s %-22s" % ("trained on", "site", "AUC", "95% CI"))
    for variant in ("full", "border_only"):
        W, b, dev_auc = heads[variant]
        ids, X = ext[variant]
        sites = np.array([parse_id(i)[0] for i in ids])
        y_all = np.array([parse_id(i)[1] for i in ids])
        p_all = R.p_tb_from_logits(R.head_logits(X, W, b), 2)
        for key, name in (("shz", "Shenzhen"), ("mcu", "Montgomery")):
            m = sites == key
            a, lo, hi = boot_auc(y_all[m], p_all[m])
            rows.append({"trained_and_tested_on": variant, "site": name,
                         "n": int(m.sum()), "n_tb": int(y_all[m].sum()),
                         "auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
                         "tbx11k_dev_auc": round(dev_auc, 4)})
            print("  %-14s %-12s %9.4f [%.4f, %.4f]" % (variant, name, a, lo, hi))

    # the dissociation
    def get(v, s):
        return next(r for r in rows if r["trained_and_tested_on"] == v and r["site"] == s)

    print("\ninterpretation:")
    verdicts = {}
    for site in ("Shenzhen", "Montgomery"):
        f, bo = get("full", site), get("border_only", site)
        internal_bo = heads["border_only"][2]
        # how much of the border shortcut's home advantage survives the move?
        kept = (bo["auc"] - 0.5) / max(internal_bo - 0.5, 1e-9)
        if bo["auc"] >= 0.80:
            v = ("SHORTCUT TRANSFERS: the external number is substantially the same "
                 "acquisition trick")
        elif bo["auc"] >= 0.65:
            v = "PARTIAL: the border carries real but minority signal at this site"
        else:
            v = ("SHORTCUT DOES NOT TRANSFER: the external result cannot be explained by "
                 "the border")
        verdicts[site] = {"border_only_auc": bo["auc"], "full_auc": f["auc"],
                          "border_above_chance_kept": round(float(kept), 4), "verdict": v}
        print("  %-11s full %.4f | border-only %.4f (keeps %.0f%% of its home advantage)"
              % (site, f["auc"], bo["auc"], 100 * kept))
        print("     -> %s" % v)

    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "border_frac": BORDER_FRAC,
           "tbx11k_dev_auc": {v: round(heads[v][2], 4) for v in heads},
           "external": rows, "verdicts": verdicts,
           "note": ("EXPLORATORY. Heads trained on TBX11K under each masking variant and "
                    "applied unchanged to the external images under the same variant. "
                    "internal_test untouched.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
