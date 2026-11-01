# -*- coding: utf-8 -*-
"""Is the 0.99 AUC coming from lung pathology, or from something else in the picture?

`docs/p2_prelim_result.md` flagged the headline number as a shortcut-learning risk and then
nothing tested it. This is the test, and it is the decisive form of it: train the identical
head on images with the **lungs removed**.

Three image variants, all from the real C05 lung masks:

  - `full`        the ordinary image (reference)
  - `lungs_only`  everything outside the lung mask replaced by a constant
  - `lungs_erased` the lung field replaced by a constant; only the periphery survives

The logic is a dissociation rather than a single number:

  - If `lungs_only` retains the accuracy, the model is reading lung tissue. Good.
  - **If `lungs_erased` also retains it, the model can diagnose TB from an image containing
    no lungs**, which means a large part of the reported accuracy is a shortcut — markers,
    collimation, patient positioning, scanner rendering, anything that correlates with which
    folder the image came from.

A constant fill is used rather than noise or blur because noise is itself informative (its
boundary traces the lung outline) and blur leaves low-frequency pathology in place. The fill
value is the per-image mean, so the masked region does not announce itself as unusually
bright or dark.

    py -3 scripts/p2_19_shortcut_probe.py --stage all

Stages are cached; embedding each variant takes about 14 minutes on the laptop GPU.
"""
import argparse
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

MANIFEST = ROOT / "data/processed/v1/manifest.parquet"
ROI = ROOT / "data/processed/v1/lung_roi.parquet"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
OUT_DIR = ROOT / "artifacts/p2/shortcut"
REF_EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
SEEDS = (17, 29, 43, 61, 79)
SPEC_SELECT = 0.95
SPEC_REPORT = 0.70
N_BOOT = 2000
VARIANTS = ("lungs_only", "lungs_erased", "lungs_erased_dilated", "border_only")
DILATE_PX = 48          # ~10% of 512: removes apices and peri-hilar margin with room to spare
BORDER_FRAC = 0.12      # outer 12% frame only


def masked_image(img, mask, mode):
    """Replace part of the image with its own mean. Mask is C05's uint8 PNG; >127 is lung.

    `lungs_erased` removes the segmented lung field, but a segmenter trims the **apices**,
    and TB is classically an upper-lobe disease — so that variant can still contain real
    pathology. `lungs_erased_dilated` erases the lung mask grown by DILATE_PX, which removes
    the apices and the peri-hilar margin too. `border_only` keeps just an outer frame, which
    contains no lung at any plausible segmentation error. The three together separate "the
    mask leaked some lung" from "the signal is not in the lungs at all".
    """
    import cv2
    m = mask > 127
    if m.sum() == 0 or (~m).sum() == 0:
        return None                     # degenerate mask: skip rather than fabricate
    out = img.astype(np.float32).copy()
    fill = float(out.mean())
    if mode == "lungs_only":
        out[~m] = fill
    elif mode == "lungs_erased":
        out[m] = fill
    elif mode == "lungs_erased_dilated":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1,) * 2)
        grown = cv2.dilate(m.astype(np.uint8), k) > 0
        if (~grown).sum() < 0.05 * m.size:
            return None                 # dilation swallowed the image; nothing left to read
        out[grown] = fill
    elif mode == "border_only":
        h, w = out.shape
        bh, bw = int(round(h * BORDER_FRAC)), int(round(w * BORDER_FRAC))
        keep = np.zeros_like(m)
        keep[:bh, :] = keep[-bh:, :] = True
        keep[:, :bw] = keep[:, -bw:] = True
        out[~keep] = fill
    else:
        raise ValueError("unknown mode %r" % mode)
    return np.clip(out, 0, 255).astype(np.uint8)


def stage_embed(args, log=print):
    """Embed both masked variants with the same backbone and settings as every other run."""
    import cv2
    from PIL import Image
    import pandas as pd

    variant = E.parse_variant("full_clahe0")
    df, _ = E.load_manifest(MANIFEST, variant)
    roi = pd.read_parquet(ROI).drop_duplicates("image_id").set_index("image_id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in VARIANTS:
        out = OUT_DIR / ("radino_tbx11k_%s_v1.npz" % mode)
        if out.exists() and not args.force:
            log("cached: %s" % out.name)
            continue
        emb = E.Embedder(batch=args.batch)
        ids, vecs, skipped = [], [], 0
        rows = list(df.itertuples(index=False))
        t0 = time.perf_counter()
        buf_imgs, buf_ids = [], []

        def flush():
            if buf_imgs:
                vecs.append(emb.embed(buf_imgs))
                ids.extend(buf_ids)
                buf_imgs.clear()
                buf_ids.clear()

        for k, r in enumerate(rows):
            if r.image_id not in roi.index:
                skipped += 1
                continue
            try:
                img = E.read_image(r.src_path, None)
                mp = ROOT / str(roi.loc[r.image_id, "mask_path"])
                mask = np.array(Image.open(mp).convert("L"), np.uint8)
                if mask.shape != img.shape:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                a = masked_image(img, mask, mode)
            except Exception as e:
                log("  skipped %s: %s" % (r.image_id, e))
                skipped += 1
                continue
            if a is None:
                skipped += 1
                continue
            buf_imgs.append(a)
            buf_ids.append(r.image_id)
            if len(buf_imgs) >= args.batch:
                flush()
            if k % 2000 == 0 and k:
                log("  %s %5d/%d  %.1f img/s"
                    % (mode, k, len(rows), k / max(1e-9, time.perf_counter() - t0)))
        flush()
        arr = np.concatenate(vecs)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:          # savez would append .npz to the path
            np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U32"), emb=arr)
        import os
        os.replace(tmp, out)
        log("wrote %s (%d images, %d skipped, %.1f min)"
            % (out.name, len(ids), skipped, (time.perf_counter() - t0) / 60))
        del emb


def load(path):
    with np.load(path) as z:
        return z["image_id"].tolist(), z["emb"].astype(np.float32)


def splits_for(ids):
    import pandas as pd
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    out = {}
    for sp in ("train", "development"):
        g = c01[c01.split == sp]
        out[sp] = ([idx[i] for i in g.image_id], H.binary_labels(g.category.tolist()))
    return out


def evaluate(label, emb, splits, log=print):
    (tr, ytr), (dv, ydv) = splits["train"], splits["development"]
    res, mean_auc, sd = H.train_seeds(
        emb[tr], ytr, emb[dv], ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    W = res[0].state_dict["weight"].numpy().astype(np.float32)
    b = res[0].state_dict["bias"].numpy().astype(np.float32)
    p = R.p_tb_from_logits(R.head_logits(emb[dv], W, b), 2)
    s95, _thr, n_pos, tp = SEL.sensitivity_at_specificity(ydv, p, SPEC_SELECT)
    s70 = SEL.sensitivity_at_specificity(ydv, p, SPEC_REPORT)[0]
    lo, hi = SEL.wilson_ci(tp, n_pos)
    log("  %-14s AUC %.4f (seed sd %.4f) | sens@95 %.4f [%.4f, %.4f] | sens@70 %.4f"
        % (label, mean_auc, sd, s95, lo, hi, s70))
    return {"variant": label, "auc": round(float(mean_auc), 4),
            "seed_sd": round(float(sd), 4), "sens_at_95": round(float(s95), 4),
            "sens_at_95_ci": [round(lo, 4), round(hi, 4)],
            "sens_at_70": round(float(s70), 4), "_p": p, "_y": ydv}


def paired(y, p_a, p_b, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        d.append(SEL.auc(y[k], p_a[k]) - SEL.auc(y[k], p_b[k]))
    d = np.asarray(d)
    return (float(SEL.auc(y, p_a) - SEL.auc(y, p_b)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def stage_compare(log=print):
    paths = {"full": REF_EMB}
    for m in VARIANTS:
        p = OUT_DIR / ("radino_tbx11k_%s_v1.npz" % m)
        if p.exists():
            paths[m] = p
    missing = [m for m in VARIANTS if m not in paths]
    if missing:
        raise SystemExit("not embedded yet: %s (run --stage embed)" % ", ".join(missing))

    # restrict every variant to the same image set so the bootstrap is paired
    sets = {k: load(v) for k, v in paths.items()}
    common = set(sets["full"][0])
    for k in VARIANTS:
        common &= set(sets[k][0])
    order = [i for i in sets["full"][0] if i in common]
    log("%d images present in all three variants" % len(order))

    aligned = {}
    for k, (ids, emb) in sets.items():
        idx = {i: j for j, i in enumerate(ids)}
        aligned[k] = emb[[idx[i] for i in order]]
    splits = splits_for(order)
    log("train %d | development %d (%d TB)\n"
        % (len(splits["train"][0]), len(splits["development"][0]),
           int(splits["development"][1].sum())))

    log("identical head, seeds %s:" % list(SEEDS))
    order_v = ["full"] + [v for v in VARIANTS if v in aligned]
    rows = [evaluate(k, aligned[k], splits) for k in order_v]
    ref = rows[0]
    ydv = ref["_y"]

    log("\npaired bootstrap vs the full image:")
    for r in rows[1:]:
        d, lo, hi = paired(ydv, r["_p"], ref["_p"])
        r["delta_auc"] = round(d, 4)
        r["ci95"] = [round(lo, 4), round(hi, 4)]
        r["verdict"] = "worse" if hi < 0 else "better" if lo > 0 else "no difference"
        log("  %-14s %+.4f [%+.4f, %+.4f]  %s" % (r["variant"], d, lo, hi, r["verdict"]))

    by = {r["variant"]: r for r in rows}

    def retained(v):
        return ((by[v]["auc"] - 0.5) / max(ref["auc"] - 0.5, 1e-9)) if v in by else None

    log("\nabove-chance AUC retained, relative to the full image:")
    for v in VARIANTS:
        if v in by:
            log("  %-22s %.4f  ->  %5.1f%%" % (v, by[v]["auc"], 100 * retained(v)))

    log("\ninterpretation:")
    if "lungs_only" in by and by["lungs_only"]["auc"] >= ref["auc"] - 0.01:
        log("  lung tissue alone reproduces the full-image result, as it should.")
    # the dissociation: a segmenter trims the apices, so lungs_erased alone is ambiguous
    strongest = None
    for v in ("border_only", "lungs_erased_dilated", "lungs_erased"):
        if v in by:
            strongest = v
            break
    if strongest:
        a = by[strongest]["auc"]
        log("  strongest lung-free test available: %s at AUC %.4f" % (strongest, a))
        if a >= 0.90:
            log("  -> SHORTCUT: TB is predictable with no lung field in the picture.")
        elif a >= 0.75:
            log("  -> SUBSTANTIAL SHORTCUT: non-lung regions carry most of the signal.")
        elif a >= 0.65:
            log("  -> PARTIAL SHORTCUT: non-lung regions carry real but minority signal.")
        else:
            log("  -> no large shortcut detected outside the lungs.")
    if "lungs_erased" in by and "lungs_erased_dilated" in by:
        drop = by["lungs_erased"]["auc"] - by["lungs_erased_dilated"]["auc"]
        log("  dilating the erased region by %d px costs %.4f AUC; a large drop means the "
            "plain\n  erase was leaking apical lung, a small one means it was not."
            % (DILATE_PX, drop))

    out = {"created": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS),
           "n_images": len(order), "n_dev": len(splits["development"][0]),
           "n_dev_tb": int(splits["development"][1].sum()),
           "fill": "per-image mean", "mask_source": "C05 lung masks",
           "dilate_px": DILATE_PX, "border_frac": BORDER_FRAC,
           "variants": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
           "above_chance_auc_retained": {v: (round(retained(v), 4) if v in by else None)
                                         for v in VARIANTS},
           "note": ("Provisional split, development only; internal_test untouched. "
                    "lungs_erased contains no lung field at all.")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "shortcut_probe.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    log("\nwrote %s" % p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("embed", "compare", "all"), default="all")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("embed", "all"):
        stage_embed(args)
    if args.stage in ("compare", "all"):
        stage_compare()


if __name__ == "__main__":
    sys.exit(main())
