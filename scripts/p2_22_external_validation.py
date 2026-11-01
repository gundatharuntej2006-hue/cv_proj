# -*- coding: utf-8 -*-
"""External validation on Shenzhen and Montgomery: the number the project has been missing.

Every result so far is single-source TBX11K, and every document says so as a limitation.
It turns out that limitation was removable all along, for reasons worth recording:

  - TBX11K's **benchmark** (11,200 images) is entirely self-collected Chinese hospital data.
    Shenzhen and Montgomery are *not* part of it.
  - But the distributed archive ships an `extra/` directory holding 327 Shenzhen, 73
    Montgomery and 176 DA/DB images, referenced **only** by the `all_*.txt` lists. Train on
    those lists and external images silently enter training.
  - This project's manifest uses the **core** lists (`TBX11K_trainval.txt`, 8,400 images), so
    none of them did. Verified empirically rather than assumed: the maximum cosine similarity
    between any of the 800 NLM images and any of the 8,400 training embeddings is **0.8389**,
    with zero pairs above 0.90.

So Shenzhen (336 TB / 326 normal) and Montgomery (58 TB / 80 normal) are genuinely unseen,
and both carry their own controls from the same hospital and programme — which matters,
because a set whose positives and negatives come from different sources can be "solved" by
recognising the acquisition pipeline rather than the disease.

**Preprocessing.** The external images are raw (3000x2919, 4892x4020, some palette-mode),
while every TBX11K image passed through C04's 512x512 standardisation. Embedding them raw
would measure that mismatch instead of the model, so they go through the identical
`cv2.resize(..., (512,512), INTER_AREA)` step first.

**Status: EXPLORATORY, not reportable.** Running an external set repeatedly during
development is exactly what P2-14's seals exist to prevent. The reportable external number
must be a sealed run of the final model on the final C01 (`docs/p2_sealed_scoring.md`). This
script exists to find out whether the system works off its home dataset at all, and its
output is labelled accordingly.

    py -3 scripts/p2_22_external_validation.py
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
from p2_gate import ood as O                                        # noqa: E402

FLAT = ROOT / "data/raw/_external_flat"
TB_EMB = ROOT / "artifacts/p2/emb/radino_tbx11k_full_clahe0_c04_v1.npz"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
BUNDLE = ROOT / "artifacts/p2/bundle"
OUT_EMB = ROOT / "artifacts/p2/external/radino_nlm_512_v1.npz"
OUT = ROOT / "artifacts/p2/external/external_validation.json"
SEEDS = (17, 29, 43, 61, 79)
SITES = {"shz": "Shenzhen", "mcu": "Montgomery"}
N_BOOT = 2000


def standardise_and_embed(force=False, batch=16, log=print):
    """The 800 NLM images through the exact C04 512px path, then the usual embedder."""
    if OUT_EMB.exists() and not force:
        with np.load(OUT_EMB) as z:
            return z["image_id"].tolist(), z["emb"].astype(np.float32)
    import cv2
    from PIL import Image
    paths = sorted(p for p in FLAT.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise SystemExit("no images in %s" % FLAT)
    emb = E.Embedder(batch=batch)
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
        buf.append(a)
        bids.append(p.stem)
        if len(buf) >= batch:
            flush()
    flush()
    arr = np.concatenate(vecs)
    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_EMB.with_suffix(".tmp")
    with open(tmp, "wb") as fh:            # savez would append .npz to the path
        np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U48"), emb=arr)
    import os
    os.replace(tmp, OUT_EMB)
    log("embedded %d external images through the C04 512px path (%.1f min)"
        % (len(ids), (time.perf_counter() - t0) / 60))
    return ids, arr


def parse_id(s):
    """'shz__CHNCXR_0512_1' -> ('shz', 1). The NLM sets encode the label in the filename."""
    site, _, rest = s.partition("__")
    return site, int(rest.rsplit("_", 1)[1])


def boot_auc(y, p, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    vals = []
    for _ in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        vals.append(SEL.auc(y[k], p[k]))
    return (float(SEL.auc(y, p)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    import pandas as pd
    meta = json.load(open(BUNDLE / "bundle.json", encoding="utf-8"))
    T = meta.get("temperature") or 1.0
    t_lower = meta.get("t_lower")
    ood_thr = meta.get("ood_threshold")

    # 1. the frozen model, trained exactly as everywhere else
    with np.load(TB_EMB) as z:
        tb_ids, TB = z["image_id"].tolist(), z["emb"].astype(np.float32)
    idx = {i: k for k, i in enumerate(tb_ids)}
    c01 = pd.read_csv(SPLITS)
    c01 = c01[c01.image_id.isin(idx)]
    tr, dv = c01[c01.split == "train"], c01[c01.split == "development"]
    Xtr, ytr = TB[[idx[i] for i in tr.image_id]], H.binary_labels(tr.category.tolist())
    Xdv, ydv = TB[[idx[i] for i in dv.image_id]], H.binary_labels(dv.category.tolist())
    res, dev_auc, sd = H.train_seeds(
        Xtr, ytr, Xdv, ydv,
        H.TrainConfig(head_type="binary", imbalance="logit_adjusted", max_epochs=60),
        seeds=SEEDS)
    W = res[0].state_dict["weight"].numpy().astype(np.float32)
    b = res[0].state_dict["bias"].numpy().astype(np.float32)
    print("frozen head: TBX11K development AUC %.4f (seed sd %.4f)" % (dev_auc, sd))

    def cal(X):
        p = R.p_tb_from_logits(R.head_logits(X, W, b), 2)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))

    # 2. the external images
    ex_ids, EX = standardise_and_embed()
    sites = np.array([parse_id(i)[0] for i in ex_ids])
    y_ex = np.array([parse_id(i)[1] for i in ex_ids])
    p_ex = cal(EX)
    p_dv = cal(Xdv)

    # the gate, as it would actually behave
    with np.load(BUNDLE / "bank.npz") as z:
        bank = O.OodBank(z["bank"].astype(np.float32), k=meta.get("ood_k", 5))
    s_ood = bank.score(EX)

    rows = []
    print("\n%-12s %5s %5s %9s %-22s %13s %13s %10s"
          % ("site", "n", "TB", "AUC", "95% CI", "sens@t_lower", "spec@t_lower", "gate rej"))
    for key, name in SITES.items():
        m = sites == key
        if m.sum() == 0:
            continue
        y, p, s = y_ex[m], p_ex[m], s_ood[m]
        a, lo, hi = boot_auc(y, p)
        sens = float((p[y == 1] >= t_lower).mean())
        spec = float((p[y == 0] < t_lower).mean())
        rej = float((s > ood_thr).mean()) if ood_thr is not None else float("nan")
        # what it could do with local recalibration
        s95 = SEL.sensitivity_at_specificity(y, p, 0.95)
        s70 = SEL.sensitivity_at_specificity(y, p, 0.70)
        rows.append({"site": name, "n": int(m.sum()), "n_tb": int(y.sum()),
                     "auc": round(a, 4), "auc_ci95": [round(lo, 4), round(hi, 4)],
                     "sens_at_tbx11k_threshold": round(sens, 4),
                     "spec_at_tbx11k_threshold": round(spec, 4),
                     "gate_rejection_rate": round(rej, 4),
                     "sens_at_95_spec_local": round(float(s95[0]), 4),
                     "sens_at_70_spec_local": round(float(s70[0]), 4),
                     "median_p_tb": round(float(np.median(p[y == 1])), 4),
                     "median_p_normal": round(float(np.median(p[y == 0])), 4)})
        print("%-12s %5d %5d %9.4f [%.4f, %.4f] %13.4f %13.4f %9.1f%%"
              % (name, m.sum(), y.sum(), a, lo, hi, sens, spec, 100 * rej))

    # 3. the honest framing of the domain gap: can a classifier tell the sources apart?
    print("\ndomain gap (TBX11K development vs each external site):")
    gap = {}
    for key, name in SITES.items():
        m = sites == key
        y_src = np.r_[np.zeros(len(Xdv), int), np.ones(int(m.sum()), int)]
        # the OOD score alone, as a source detector
        s_src = np.r_[bank.score(Xdv), s_ood[m]]
        gap[name] = {"source_auroc_from_ood_score": round(float(SEL.auc(y_src, s_src)), 4),
                     "median_ood_tbx11k": round(float(np.median(bank.score(Xdv))), 4),
                     "median_ood_site": round(float(np.median(s_ood[m])), 4)}
        print("  %-12s OOD score separates sources at AUROC %.4f (median %.4f vs %.4f)"
              % (name, gap[name]["source_auroc_from_ood_score"],
                 gap[name]["median_ood_tbx11k"], gap[name]["median_ood_site"]))

    # 4. how much did we lose by leaving home?
    dev_a, dev_lo, dev_hi = boot_auc(ydv, p_dv)
    print("\nTBX11K development AUC %.4f [%.4f, %.4f]" % (dev_a, dev_lo, dev_hi))
    for r in rows:
        print("  %-12s %.4f -> drop %+.4f" % (r["site"], r["auc"], r["auc"] - dev_a))

    out = {"created": time.strftime("%Y-%m-%d"), "status": "EXPLORATORY, not reportable",
           "why": ("the reportable external number must be a sealed run of the final model "
                   "on the final C01; see docs/p2_sealed_scoring.md"),
           "model": {"dev_auc": round(float(dev_auc), 4), "seed_sd": round(float(sd), 4),
                     "variant": meta["variant"], "temperature": T, "t_lower": t_lower,
                     "ood_threshold": ood_thr},
           "tbx11k_development": {"auc": round(dev_a, 4),
                                  "auc_ci95": [round(dev_lo, 4), round(dev_hi, 4)],
                                  "n": int(len(ydv)), "n_tb": int(ydv.sum())},
           "external_sites": rows, "domain_gap": gap,
           "leakage_check": {"max_cosine_external_vs_train": 0.8389,
                             "pairs_above_0.90": 0,
                             "note": "training uses the core TBX11K lists, not all_*.txt"},
           "preprocessing": "cv2.resize INTER_AREA to 512x512, matching C04",
           "note": ("Shenzhen and Montgomery each carry their own same-hospital controls, so "
                    "the comparison does not reward acquisition-pipeline detection.")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
