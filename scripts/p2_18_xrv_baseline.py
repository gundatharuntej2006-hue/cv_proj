# -*- coding: utf-8 -*-
"""P2-18: is RAD-DINO actually worth it? Compare against a published CXR CNN.

The methodology picks a frozen RAD-DINO backbone. That choice needs a baseline, otherwise
"AUC 0.99" means nothing — a DenseNet trained on 800k chest X-rays is the obvious thing a
reviewer would reach for, and it is a single pip install away.

Three comparisons, all on the same split and the same images (this script calls Person 2's
own `embed.load_manifest` / `read_image`, so the pixels going into both backbones are
byte-identical — otherwise the comparison measures preprocessing, not backbones):

  1. **Zero-shot.** TorchXRayVision's `densenet121-res224-all` predicts 18 pathologies. None
     of them is tuberculosis, so the honest zero-shot baseline is "use the best available
     proxy pathology as a TB score" — Infiltration, Consolidation, Lung Opacity and so on.
     This is what someone with no training budget would actually deploy.
  2. **Transfer.** A linear head on the DenseNet's 1024-d pooled features, trained with the
     same `heads.py` machinery, same seeds, same early stopping as the RAD-DINO head. This
     is the apples-to-apples backbone comparison.
  3. **Pathology vector.** A linear head on just the 18 pathology logits. A 18-d feature
     space that is fully interpretable; worth knowing how much is lost.

Then a paired bootstrap of each against the RAD-DINO head, and — the question that matters
more than accuracy — whether the DenseNet feature space shows the *same* pathological
within-dataset homogeneity that breaks the OOD gate (see `p2_ood_gate_finding.md`). If it
does, that failure is a property of TBX11K, not of RAD-DINO.

    py -3 scripts/p2_18_xrv_baseline.py --stage all

Stages are cached, so a crash costs only the stage it was in.
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

VARIANT = "full_clahe0"          # the variant RAD-DINO does best on (Section 14 doc)
WEIGHTS = "densenet121-res224-all"
MANIFEST = ROOT / "data/processed/v1/manifest.parquet"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
RADINO_EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
OUT_DIR = ROOT / "artifacts/p2/baseline"
XRV_EMB = OUT_DIR / "xrv_densenet_tbx11k_full_clahe0.npz"
SEEDS = (17, 29, 43, 61, 79)
SPEC_REPORT = 0.70
SPEC_SELECT = 0.95
N_BOOT = 2000


# --------------------------------------------------------------------------- feature stage
def _load_model(device):
    import io
    import contextlib
    import torchxrayvision as xrv
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        m = xrv.models.DenseNet(weights=WEIGHTS)   # its download bar crashes on cp1252
    return m.to(device).eval(), list(m.pathologies)


def _prep_batch(images):
    """uint8 512x512 -> the [-1024, 1024] 224x224 tensor TorchXRayVision expects.

    Its own normalisation helper assumes a 0-255 input, which is what `read_image` gives us.
    Getting this wrong silently halves the baseline's accuracy, which would flatter us.
    """
    import torch
    import torchxrayvision as xrv
    import cv2
    out = []
    for a in images:
        a = xrv.datasets.normalize(a.astype(np.float32), 255)       # -> [-1024, 1024]
        a = cv2.resize(a, (224, 224), interpolation=cv2.INTER_AREA)
        out.append(a[None])
    return torch.from_numpy(np.stack(out)).float()


def stage_features(args, log=print):
    """Extract 1024-d pooled features + 18 pathology probabilities for every image."""
    if XRV_EMB.exists() and not args.force:
        log("features cached: %s" % XRV_EMB)
        return
    import torch
    import torch.nn.functional as F

    variant = E.parse_variant(VARIANT)
    df, _preproc = E.load_manifest(
        MANIFEST, variant, roi_path=args.roi if variant.kind == "crop" else None)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, pathologies = _load_model(device)
    log("xrv %s on %s | %d images" % (WEIGHTS, device, len(df)))

    ids, feats, probs = [], [], []
    t0 = time.perf_counter()
    rows = list(df.itertuples(index=False))
    for s in range(0, len(rows), args.batch):
        chunk = rows[s:s + args.batch]
        imgs, keep = [], []
        for r in chunk:
            try:
                a = E.read_image(r.src_path, getattr(r, "crop_box", None))
            except Exception as e:          # one bad file must not end an 8,400-image run
                log("  skipped %s: %s" % (r.image_id, e))
                continue
            imgs.append(a)
            keep.append(r.image_id)
        if not imgs:
            continue
        x = _prep_batch(imgs).to(device)
        with torch.no_grad():
            fm = model.features(x)                      # [N, 1024, 7, 7]
            pooled = F.adaptive_avg_pool2d(fm, 1).flatten(1)
            p = model(x)                                # [N, 18] sigmoid outputs
        ids.extend(keep)
        feats.append(pooled.cpu().numpy().astype(np.float32))
        probs.append(p.cpu().numpy().astype(np.float32))
        if (s // args.batch) % 50 == 0:
            done = len(ids)
            log("  %5d/%d  %.1f img/s" % (done, len(rows), done / max(1e-9, time.perf_counter() - t0)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = XRV_EMB.with_suffix(".tmp")
    with open(tmp, "wb") as fh:                         # np.savez would append .npz
        np.savez_compressed(fh, image_id=np.array(ids), feat=np.concatenate(feats),
                            prob=np.concatenate(probs),
                            pathologies=np.array(pathologies))
    import os
    os.replace(tmp, XRV_EMB)
    log("wrote %s (%d images, %.1f min)" % (XRV_EMB, len(ids), (time.perf_counter() - t0) / 60))


# --------------------------------------------------------------------------- shared loading
def _splits(ids, extra_cols=()):
    import pandas as pd
    idx = {i: k for k, i in enumerate(ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    out = {}
    for sp in ("train", "development", "calibration"):
        g = c01[c01.split == sp]
        out[sp] = ([idx[i] for i in g.image_id], H.binary_labels(g.category.tolist()))
    return out


def _train_and_score(X, splits, label, log=print):
    """Same head, seeds and criterion as the RAD-DINO run, so only the features differ."""
    (tr, ytr), (dv, ydv) = splits["train"], splits["development"]
    res, mean_auc, sd = H.train_seeds(
        X[tr], ytr, X[dv], ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    W = res[0].state_dict["weight"].numpy().astype(np.float32)
    b = res[0].state_dict["bias"].numpy().astype(np.float32)
    p = R.p_tb_from_logits(R.head_logits(X[dv], W, b), 2)
    s70 = SEL.sensitivity_at_specificity(ydv, p, SPEC_REPORT)[0]
    s95 = SEL.sensitivity_at_specificity(ydv, p, SPEC_SELECT)[0]
    log("  %-22s dev AUC %.4f (sd %.4f) | sens@70 %.4f | sens@95 %.4f"
        % (label, mean_auc, sd, s70, s95))
    return {"label": label, "dim": int(X.shape[1]), "auc": round(float(mean_auc), 4),
            "seed_sd": round(float(sd), 4),
            "sens_at_70": round(float(s70), 4), "sens_at_95": round(float(s95), 4),
            "_p_dev": p, "_y_dev": ydv}


def _paired_bootstrap(y, p_a, p_b, n_boot=N_BOOT, seed=0):
    """ΔAUC = a - b with a paired CI. Same resampled cases for both, so the comparison is
    paired and the CI is far tighter than two independent intervals."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        if len(set(y[k].tolist())) < 2:
            continue
        d.append(SEL.auc(y[k], p_a[k]) - SEL.auc(y[k], p_b[k]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (float(SEL.auc(y, p_a) - SEL.auc(y, p_b)), float(lo), float(hi))


# --------------------------------------------------------------------------- site geometry
CONTROL_DIR = ROOT / "data/raw/ood/_control_frontal_512"
XRV_CONTROL = OUT_DIR / "xrv_densenet_control_frontal.npz"


def _xrv_embed_dir(directory, out_path, args, log=print):
    """Embed a directory of 512px frontals with the DenseNet, cached."""
    if out_path.exists() and not args.force:
        with np.load(out_path) as z:
            return z["feat"].astype(np.float32)
    import torch
    import torch.nn.functional as F
    from PIL import Image
    paths = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise SystemExit("no images in %s" % directory)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model(device)
    feats = []
    for s in range(0, len(paths), args.batch):
        imgs = [np.array(Image.open(p).convert("L"), np.uint8) for p in paths[s:s + args.batch]]
        x = _prep_batch(imgs).to(device)
        with torch.no_grad():
            feats.append(F.adaptive_avg_pool2d(model.features(x), 1).flatten(1).cpu().numpy())
    out = np.concatenate(feats).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        np.savez_compressed(fh, feat=out)
    log("  embedded %d control images with the DenseNet" % len(out))
    return out


def _knn_mean_distance(bank, query, k=5):
    """The D13 OOD score: mean cosine distance to the k nearest bank points."""
    B = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-9)
    Q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-9)
    out = np.empty(len(Q), np.float32)
    for s in range(0, len(Q), 256):
        d = 1.0 - (Q[s:s + 256] @ B.T)
        out[s:s + 256] = np.sort(d, axis=1)[:, :k].mean(axis=1)
    return out


def stage_site(args, remb, order, log=print):
    """Is TBX11K's pathological homogeneity a property of RAD-DINO or of the dataset?

    A raw within-dataset distance is **not** comparable across backbones: DenseNet features
    are post-ReLU and therefore non-negative, so every cosine similarity is high and every
    distance small by construction, regardless of how much real variation there is. The
    scale-free quantity is the **ratio** of the distance a new site sits at to the distance
    the dataset's own images sit at — and the site separability AUROC, which is invariant to
    any monotone rescaling of the score.
    """
    log("\ncross-site geometry (same k=5 OOD score as D13, both spaces):")
    xfeat_ctrl = _xrv_embed_dir(CONTROL_DIR, XRV_CONTROL, args, log=log)
    rctrl_p = ROOT / "artifacts/p2/emb/radino_control_frontal512_v1.npz"
    if not rctrl_p.exists():
        log("  (no RAD-DINO control embeddings; skipping)")
        return {}
    with np.load(rctrl_p) as z:
        rfeat_ctrl = z["emb"].astype(np.float32)

    xi = {i: k for k, i in enumerate(order)}
    rng = np.random.default_rng(0)
    with np.load(XRV_EMB) as z:
        xids = z["image_id"].tolist()
        xall = z["feat"].astype(np.float32)
    xmap = {i: k for k, i in enumerate(xids)}
    xfeat = xall[[xmap[i] for i in order]]

    out = {}
    n_bank = min(4000, len(order))
    bank_idx = rng.choice(len(order), n_bank, replace=False)
    held = np.setdiff1d(np.arange(len(order)), bank_idx)[:1000]
    for name, X, ctrl in (("rad-dino", remb, rfeat_ctrl), ("xrv densenet", xfeat, xfeat_ctrl)):
        bank = X[bank_idx]
        d_self = _knn_mean_distance(bank, X[held])
        d_new = _knn_mean_distance(bank, ctrl)
        y = np.r_[np.zeros(len(d_self), int), np.ones(len(d_new), int)]
        auroc = float(SEL.auc(y, np.r_[d_self, d_new]))
        ratio = float(np.median(d_new) / max(np.median(d_self), 1e-9))
        out[name] = {"median_self": round(float(np.median(d_self)), 4),
                     "median_new_site": round(float(np.median(d_new)), 4),
                     "ratio": round(ratio, 3), "site_auroc": round(auroc, 4)}
        log("  %-14s self %.4f | new site %.4f | ratio %.2fx | site AUROC %.4f"
            % (name, np.median(d_self), np.median(d_new), ratio, auroc))
    log("  (raw distances are not comparable across backbones - DenseNet features are "
        "post-ReLU and so non-negative; the ratio and the AUROC are.)")
    return out


# --------------------------------------------------------------------------- comparison
def stage_compare(args, log=print):
    if not XRV_EMB.exists():
        raise SystemExit("run --stage features first")
    with np.load(XRV_EMB, allow_pickle=False) as z:
        xids = z["image_id"].tolist()
        feat = z["feat"].astype(np.float32)
        prob = z["prob"].astype(np.float32)
        pathologies = z["pathologies"].tolist()
    with np.load(RADINO_EMB) as z:
        rids = z["image_id"].tolist()
        remb = z["emb"].astype(np.float32)

    common = [i for i in xids if i in set(rids)]
    log("%d images in both backbones" % len(common))
    xi = {i: k for k, i in enumerate(xids)}
    ri = {i: k for k, i in enumerate(rids)}
    order = common
    feat = feat[[xi[i] for i in order]]
    prob = prob[[xi[i] for i in order]]
    remb = remb[[ri[i] for i in order]]
    splits = _splits(order)
    dv, ydv = splits["development"]
    log("train %d | development %d (%d TB)"
        % (len(splits["train"][0]), len(dv), int(splits["development"][1].sum())))

    # 1. zero-shot: every pathology head as a TB proxy
    log("\nzero-shot (published pathology heads as a TB score):")
    zs = []
    for k, name in enumerate(pathologies):
        a = float(SEL.auc(ydv, prob[dv, k]))
        zs.append({"pathology": name, "auc": round(a, 4),
                   "auc_oriented": round(max(a, 1 - a), 4)})
    zs.sort(key=lambda r: -r["auc_oriented"])
    for r in zs[:6]:
        log("  %-24s AUC %.4f" % (r["pathology"], r["auc"]))
    best = zs[0]
    best_k = pathologies.index(best["pathology"])
    p_zs = prob[dv, best_k]
    if best["auc"] < 0.5:
        p_zs = -p_zs                                 # oriented; report the orientation

    # 2 & 3. trained heads
    log("\ntrained linear heads (identical head, seeds and criterion):")
    r_dino = _train_and_score(remb, splits, "rad-dino 768-d", log)
    r_feat = _train_and_score(feat, splits, "xrv densenet 1024-d", log)
    r_prob = _train_and_score(prob, splits, "xrv pathologies 18-d", log)

    # paired comparisons against RAD-DINO
    log("\npaired bootstrap, dAUC vs rad-dino (%d resamples):" % N_BOOT)
    comps = []
    for r in (r_feat, r_prob):
        d, lo, hi = _paired_bootstrap(ydv, r["_p_dev"], r_dino["_p_dev"])
        verdict = ("worse" if hi < 0 else "better" if lo > 0 else "no difference")
        comps.append({"label": r["label"], "delta_auc": round(d, 4),
                      "ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict})
        log("  %-22s %+.4f [%+.4f, %+.4f]  %s" % (r["label"], d, lo, hi, verdict))
    d, lo, hi = _paired_bootstrap(ydv, p_zs, r_dino["_p_dev"])
    comps.append({"label": "zero-shot %s" % best["pathology"], "delta_auc": round(d, 4),
                  "ci95": [round(lo, 4), round(hi, 4)],
                  "verdict": "worse" if hi < 0 else "better" if lo > 0 else "no difference"})
    log("  %-22s %+.4f [%+.4f, %+.4f]  %s"
        % ("zero-shot best", d, lo, hi, comps[-1]["verdict"]))

    # 4. is the homogeneity that breaks the OOD gate a backbone property or a dataset one?
    homo = stage_site(args, remb, order, log=log)

    out = {"created": time.strftime("%Y-%m-%d"), "variant": VARIANT, "weights": WEIGHTS,
           "n_images": len(order), "n_dev": len(dv),
           "n_dev_tb": int(splits["development"][1].sum()),
           "zero_shot": zs, "zero_shot_best": best,
           "heads": [{k: v for k, v in r.items() if not k.startswith("_")}
                     for r in (r_dino, r_feat, r_prob)],
           "paired_vs_raddino": comps,
           "within_dataset_median_distance": homo,
           "note": ("Provisional split (Person 1's C01 pending). Development split only; "
                    "internal_test untouched. Both backbones see byte-identical pixels via "
                    "p2_cls.embed.load_manifest/read_image.")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "xrv_comparison.json"
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    log("\nwrote %s" % p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("features", "compare", "site", "all"), default="all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device")
    ap.add_argument("--roi", default=str(ROOT / "data/processed/v1/lung_roi.parquet"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("features", "all"):
        stage_features(args)
    if args.stage in ("compare", "all"):
        stage_compare(args)


if __name__ == "__main__":
    sys.exit(main())
