# -*- coding: utf-8 -*-
"""Paired pooling comparison on Shenzhen and Montgomery. EXPLORATORY, not validation.

Protocol fixed in advance: artifacts/p2/audit/cross_dataset_manifest.json, committed
before any feature here was computed.

**These cohorts are not untouched.** The lungs-only masking in the configuration under test
was adopted partly because of how it scored on Montgomery and Shenzhen. Only the
*within-pair difference* between aggregators is interpreted; both arms inherit identical
contamination, so the difference survives while neither absolute number is a clean
generalisation estimate.

Three arms, all fitted on TBX11K train with healthy-only negatives, early-stopped on TBX11K
development, seeds (17, 29, 43, 61, 79). None ever sees a Shenzhen or Montgomery label.

  A  global_pooler      RAD-DINO pooler_output (the deployed global representation)
  B  patch_mean_1x1     12x12 regions averaged to 1x1 - the MATCHED control
  C  regional_max_12x12 12x12 regions, max over region logits - the candidate

B vs C isolates the aggregator exactly: same tokens, same training, same seeds, one
difference. A vs C additionally changes the representation source (CLS-pooled vs
patch-derived) and is reported for completeness, not as the clean test.

    py -3 scripts/p2_36_cross_dataset_pooling.py --stage embed
    py -3 scripts/p2_36_cross_dataset_pooling.py --stage score
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from p2_cls import heads as H                                        # noqa: E402
from p2_cls import registry as R                                     # noqa: E402
from p2_cls import selection as SEL                                  # noqa: E402

FLAT = ROOT / "data/raw/_external_flat"
MASKS = ROOT / "data/processed/v1/masks_external"
LUNGS_PNG = ROOT / "data/processed/v1/nlm_img512_lungs"
REG = ROOT / "artifacts/p2/regional"
AUDIT = ROOT / "artifacts/p2/audit"
NLM_REG = REG / "regional_nlm_lungs_g12_v1.npz"
NLM_GLOBAL = ROOT / "artifacts/p2/shortcut/radino_nlm_lungs_only_v1.npz"
TBX_REG = REG / "regional_tbx11k_lungs_g12_v1.npz"
TBX_GLOBAL = ROOT / "artifacts/p2/shortcut/radino_tbx11k_lungs_only_v1.npz"
SPLITS = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
TB = ("active_tb", "latent_tb", "active_latent_tb")
SEEDS = (17, 29, 43, 61, 79)
N_BOOT = 4000
COHORT = {"shz": "Shenzhen", "mcu": "Montgomery"}


# ------------------------------------------------------------------ eligibility + extract
def build_lungs_png(log=print):
    """Lungs-only PNGs for the NLM images, same rule as TBX11K and Vietnam. Resumable."""
    import cv2
    from PIL import Image
    LUNGS_PNG.mkdir(parents=True, exist_ok=True)
    excluded = []
    srcs = sorted(p for p in FLAT.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for p in srcs:
        dst = LUNGS_PNG / (p.stem + ".png")
        if dst.exists():
            continue
        mp = MASKS / (p.stem + ".png")
        if not mp.exists():
            excluded.append((p.stem, "no lung mask"))
            continue
        try:
            a = np.array(Image.open(p).convert("L"), dtype=np.uint8)
            if a.shape != (512, 512):
                a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
            m = np.array(Image.open(mp).convert("L"), np.uint8) > 127
        except Exception as e:
            excluded.append((p.stem, "unreadable: %s" % type(e).__name__))
            continue
        if m.sum() == 0:
            excluded.append((p.stem, "empty lung mask"))
            continue
        o = a.astype(np.float32)
        o[~m] = float(o.mean())
        Image.fromarray(np.clip(o, 0, 255).astype(np.uint8)).save(dst)
    log("  lungs-only PNGs: %d of %d source images; %d excluded"
        % (len(list(LUNGS_PNG.glob("*.png"))), len(srcs), len(excluded)))
    for s, why in excluded[:10]:
        log("    excluded %s: %s" % (s, why))
    return excluded


def stage_embed(log=print):
    import p2_31_regional_mil as RM
    if NLM_REG.exists():
        log("  cached: %s" % NLM_REG.name)
        return []
    excluded = build_lungs_png(log=log)
    from PIL import Image
    emb = RM.RegionalEmbedder(batch=16)
    if emb.g != 37:
        raise SystemExit("unexpected patch grid %d" % emb.g)
    paths = sorted(LUNGS_PNG.glob("*.png"))
    ids, vecs, buf, bids = [], [], [], []
    t0 = time.perf_counter()
    for p in paths:
        buf.append(np.array(Image.open(p).convert("L"), dtype=np.uint8))
        bids.append(p.stem)
        if len(buf) >= 16:
            vecs.append(emb.embed(buf)); ids.extend(bids); buf, bids = [], []
    if buf:
        vecs.append(emb.embed(buf)); ids.extend(bids)
    arr = np.concatenate(vecs)
    REG.mkdir(parents=True, exist_ok=True)
    tmp = NLM_REG.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, image_id=np.asarray(ids, dtype="<U48"), reg=arr)
    os.replace(tmp, NLM_REG)
    log("  embedded %d NLM images -> %s  (%.1f min)"
        % (len(ids), arr.shape, (time.perf_counter() - t0) / 60))
    return excluded


# ------------------------------------------------------------------ scoring
def boot_auc(y, p, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    v = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        v.append(SEL.auc(y[k], p[k]))
    return float(SEL.auc(y, p)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def auprc(y, p):
    y = np.asarray(y).astype(int)
    o = np.argsort(-np.asarray(p))
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / max(ys.sum(), 1)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def paired_delta(y, pa, pb, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    d = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) < 2:
            continue
        d.append(SEL.auc(y[k], pa[k]) - SEL.auc(y[k], pb[k]))
    d = np.asarray(d)
    return (float(SEL.auc(y, pa) - SEL.auc(y, pb)),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def stage_score(log=print):
    import pandas as pd
    import p2_32_mil_sweep as S

    for p in (NLM_REG, TBX_REG, NLM_GLOBAL, TBX_GLOBAL):
        if not p.exists():
            raise SystemExit("missing prerequisite: %s" % p)

    c01 = pd.read_csv(SPLITS)

    # ---- fit the three arms on TBX11K only
    with np.load(TBX_REG) as z:
        tid, TXR = z["image_id"].tolist(), z["reg"]
    ti = {i: k for k, i in enumerate(tid)}
    c = c01[c01.image_id.isin(ti)]
    tr = c[(c.split == "train") & (c.category.isin(TB + ("healthy",)))]
    dv = c[c.split == "development"]
    ytr = np.array([1 if x in TB else 0 for x in tr.category])
    ydv = np.array([1 if x in TB else 0 for x in dv.category])
    Rtr12 = S.repool(TXR[[ti[i] for i in tr.image_id]], 12, 12)
    Rdv12 = S.repool(TXR[[ti[i] for i in dv.image_id]], 12, 12)
    Rtr1 = S.repool(TXR[[ti[i] for i in tr.image_id]], 12, 1)
    Rdv1 = S.repool(TXR[[ti[i] for i in dv.image_id]], 12, 1)
    del TXR

    with np.load(TBX_GLOBAL) as z:
        gid, TXG = z["image_id"].tolist(), z["emb"].astype(np.float32)
    gi = {i: k for k, i in enumerate(gid)}
    Gtr, Gdv = TXG[[gi[i] for i in tr.image_id]], TXG[[gi[i] for i in dv.image_id]]
    del TXG

    log("fitting three arms on TBX11K train (%d, %d TB), early stop on development (%d)"
        % (len(ytr), ytr.sum(), len(ydv)))
    arms = {}
    res, dauc, _ = H.train_seeds(Gtr, ytr, Gdv, ydv,
                                 H.TrainConfig(head_type="binary",
                                               imbalance="logit_adjusted", max_epochs=60),
                                 seeds=SEEDS)
    arms["A_global_pooler"] = {"kind": "global",
                               "W": res[0].state_dict["weight"].numpy().astype(np.float32),
                               "b": res[0].state_dict["bias"].numpy().astype(np.float32),
                               "dev_auc": float(dauc)}
    for name, Xtr, Xdv, how in (("B_patch_mean_1x1", Rtr1, Rdv1, "mean"),
                                ("C_regional_max_12x12", Rtr12, Rdv12, "max")):
        Ws = [S.train(Xtr, ytr, Xdv, ydv, how, sd) for sd in SEEDS]
        arms[name] = {"kind": "mil", "how": how,
                      "heads": [(w, b) for w, b, _ in Ws],
                      "dev_auc": float(np.mean([d for _, _, d in Ws]))}
    for k, v in arms.items():
        log("  %-22s TBX11K development AUC %.4f" % (k, v["dev_auc"]))

    # ---- fixed thresholds from TBX11K development, per arm
    def dev_scores(a):
        if a["kind"] == "global":
            return R.p_tb_from_logits(R.head_logits(Gdv, a["W"], a["b"]), 2)
        X = Rdv1 if a["how"] == "mean" else Rdv12
        return np.mean([S.score(X, W, b, a["how"]) for W, b in a["heads"]], axis=0)

    thresholds = {k: float(np.quantile(dev_scores(a)[ydv == 0], 0.70)) for k, a in arms.items()}
    log("  thresholds at 70%% specificity on TBX11K development: %s"
        % {k: round(v, 4) for k, v in thresholds.items()})

    # ---- eligible NLM cases: intersection across arms
    with np.load(NLM_REG) as z:
        nrid, NR = z["image_id"].tolist(), z["reg"]
    with np.load(NLM_GLOBAL) as z:
        ngid, NG = z["image_id"].tolist(), z["emb"].astype(np.float32)
    common = [i for i in nrid if i in set(ngid)]
    only_reg = sorted(set(nrid) - set(ngid))
    only_glb = sorted(set(ngid) - set(nrid))
    log("\neligible cases: %d in both arms | regional-only %d | global-only %d"
        % (len(common), len(only_reg), len(only_glb)))
    ri = {i: k for k, i in enumerate(nrid)}
    gj = {i: k for k, i in enumerate(ngid)}
    NRc = S.repool(NR[[ri[i] for i in common]], 12, 12)
    NR1 = S.repool(NR[[ri[i] for i in common]], 12, 1)
    NGc = NG[[gj[i] for i in common]]
    del NR, NG

    y = np.array([int(str(i).rsplit("_", 1)[1]) for i in common])
    site = np.array([str(i).split("__")[0] for i in common])
    assert len(set(common)) == len(common), "duplicate image ids among eligible cases"

    preds = {}
    for k, a in arms.items():
        if a["kind"] == "global":
            preds[k] = R.p_tb_from_logits(R.head_logits(NGc, a["W"], a["b"]), 2)
        else:
            X = NR1 if a["how"] == "mean" else NRc
            preds[k] = np.mean([S.score(X, W, b, a["how"]) for W, b in a["heads"]], axis=0)

    # ---- per-cohort metrics
    rows, case_rows = [], []
    for pre, name in COHORT.items():
        m = site == pre
        yy = y[m]
        log("\n%s: n=%d  TB=%d  prevalence %.3f" % (name, m.sum(), yy.sum(), yy.mean()))
        for k in arms:
            pp = preds[k][m]
            a, lo, hi = boot_auc(yy, pp)
            thr = thresholds[k]
            pred = pp >= thr
            tp, fn = int((pred & (yy == 1)).sum()), int((~pred & (yy == 1)).sum())
            fp, tn = int((pred & (yy == 0)).sum()), int((~pred & (yy == 0)).sum())
            rows.append({"cohort": name, "arm": k, "n": int(m.sum()), "n_tb": int(yy.sum()),
                         "prevalence": round(float(yy.mean()), 4),
                         "auroc": round(a, 4), "auroc_lo": round(lo, 4),
                         "auroc_hi": round(hi, 4), "auprc": round(auprc(yy, pp), 4),
                         "threshold_from_tbx11k_dev": round(thr, 6),
                         "sensitivity": round(tp / max(tp + fn, 1), 4),
                         "specificity": round(tn / max(tn + fp, 1), 4),
                         "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                         "tbx11k_dev_auc": round(arms[k]["dev_auc"], 4)})
            log("  %-22s AUROC %.4f [%.4f, %.4f]  AUPRC %.4f  sens %.3f spec %.3f"
                % (k, a, lo, hi, auprc(yy, pp), tp / max(tp + fn, 1), tn / max(tn + fp, 1)))
        # paired differences
        for ka, kb, label in (("C_regional_max_12x12", "B_patch_mean_1x1", "C-B (pure pooling)"),
                              ("C_regional_max_12x12", "A_global_pooler", "C-A (pooling+repr)")):
            d, lo, hi = paired_delta(yy, preds[ka][m], preds[kb][m])
            rows.append({"cohort": name, "arm": "DELTA " + label, "n": int(m.sum()),
                         "n_tb": int(yy.sum()), "prevalence": round(float(yy.mean()), 4),
                         "auroc": round(d, 4), "auroc_lo": round(lo, 4),
                         "auroc_hi": round(hi, 4), "auprc": None,
                         "threshold_from_tbx11k_dev": None, "sensitivity": None,
                         "specificity": None, "tp": None, "fn": None, "fp": None,
                         "tn": None, "tbx11k_dev_auc": None})
            verdict = "worse" if hi < 0 else "better" if lo > 0 else "no difference"
            log("  %-22s dAUROC %+.4f [%+.4f, %+.4f]  %s" % (label, d, lo, hi, verdict))

    for i, idx in enumerate(common):
        case_rows.append({"image_id": idx, "cohort": COHORT[site[i]], "label": int(y[i]),
                          **{("score_" + k): round(float(preds[k][i]), 6) for k in arms}})

    import pandas as pd
    AUDIT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AUDIT / "cross_dataset_metrics.csv", index=False)
    pd.DataFrame(case_rows).to_csv(AUDIT / "cross_dataset_case_predictions.csv", index=False)
    json.dump({"created": time.strftime("%Y-%m-%d"), "manifest": "cross_dataset_manifest.json",
               "eligible": len(common), "regional_only": only_reg, "global_only": only_glb,
               "thresholds": thresholds,
               "tbx11k_dev_auc": {k: round(v["dev_auc"], 4) for k, v in arms.items()},
               "classification": "EXPLORATORY PAIRED BENCHMARK, not external validation"},
              open(AUDIT / "cross_dataset_run.json", "w", encoding="utf-8"), indent=1)
    log("\nwrote cross_dataset_metrics.csv, cross_dataset_case_predictions.csv, cross_dataset_run.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("embed", "score", "all"), default="all")
    a = ap.parse_args()
    if a.stage in ("embed", "all"):
        stage_embed()
    if a.stage in ("score", "all"):
        stage_score()


if __name__ == "__main__":
    sys.exit(main())
