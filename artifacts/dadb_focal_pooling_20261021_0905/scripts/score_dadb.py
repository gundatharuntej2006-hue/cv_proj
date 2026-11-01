# -*- coding: utf-8 -*-
"""Phase 1b + Phase 2: fit both arms on TBX11K, freeze the manifest, score DA/DB once.

Arms, both fitted on TBX11K train only (healthy-only negatives), early-stopped on TBX11K
development, seeds 17/29/43/61/79 -- the configuration of
`scripts/p2_36_cross_dataset_pooling.py`:

  A  global_pooler        RAD-DINO pooler_output over the lungs-only image
  B  regional_max_12x12   12x12 region logits, max-pooled

Two reporting variants, both declared here before any DA/DB score exists:

  PRIMARY   both arms use the 5-seed ensemble. This is the matched comparison the task
            asks for -- identical seed treatment on both sides.
  SECONDARY arm A uses seed 17 alone, arm B the 5-seed ensemble. This replicates
            `p2_36` exactly, which took `res[0]` for the global arm, and exists only so
            the DA/DB numbers sit on the same footing as the published Shenzhen and
            Montgomery ones.

No DA/DB label is read until every head and threshold is frozen, and none is used to fit,
tune, threshold, select a seed or choose preprocessing.
"""
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import common as C                                                   # noqa: E402

TBX_REG = C.REG / "regional_tbx11k_lungs_g12_v1.npz"
TBX_GLOBAL = ROOT / "artifacts/p2/shortcut/radino_tbx11k_lungs_only_v1.npz"
DADB_REG = RUN / "tables/regional_dadb_lungs_g12_v1.npz"
DADB_GLOBAL = ROOT / "artifacts/p2/external/radino_dadb_lungs_only_v1.npz"
SRC = ROOT / "data/raw/tbx11k/TBX11K/imgs/extra/da+db"

SEEDS = (17, 29, 43, 61, 79)
TARGET_SPEC = 0.70
N_BOOT = 4000
BOOT_SEED = 0
SANITY_FAIL_ID = "px4"            # identified label-blind in Phase 0; reason "components"

# published TBX11K development AUCs from p2_36, used as an end-to-end equivalence check
EXPECT_DEV = {"A_global_pooler": 0.9570, "B_regional_max_12x12": 0.9368}

T0 = time.perf_counter()


def log(m):
    print("[%6.1fs] %s" % (time.perf_counter() - T0, m), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dadb_manifest():
    """case_id, site, label -- exactly p2_26::images(). Labels used for SCORING only."""
    rows = []
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        rows.append({"order_index": len(rows), "case_id": p.stem,
                     "site": "DA" if p.suffix.lower() == ".png" else "DB",
                     "label": 1 if p.name[0] == "p" else 0})
    return rows


def prove_streaming_equivalence(dev):
    """Required before bulk use: streamed tensor vs committed repool, on a fixed sample."""
    import p2_32_mil_sweep as S
    idx = np.arange(64)
    with np.load(DADB_REG) as z:
        raw = np.asarray(z["reg"][:64])
    ref = S.repool(raw, 12, 12)
    got_t, kept = C.stream_regions(DADB_REG, keep_idx=idx, block=16, device=dev,
                                   verbose=False)
    got = got_t.cpu().numpy()
    assert list(kept) == list(idx)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    ident = int((got == ref).sum())
    out = {"sample_n": 64, "shape": list(ref.shape),
           "elementwise_identical": ident, "elements": int(ref.size),
           "pct_identical": round(100.0 * ident / ref.size, 6),
           "max_abs_difference": float(diff.max()),
           "mean_abs_difference": float(diff.mean())}
    log("equivalence: %.6f%% identical, max abs diff %.3e"
        % (out["pct_identical"], out["max_abs_difference"]))
    if diff.max() >= 1e-3:
        raise SystemExit("streaming path diverges from the committed repool")
    del got_t
    return out


def main():
    import torch
    import pandas as pd
    from p2_cls import heads as H
    from p2_cls import registry as R

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device %s" % dev)

    equiv = prove_streaming_equivalence(dev)
    torch.cuda.empty_cache()

    # ---------------- TBX11K training rows ----------------
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    tids = C.read_ids(TBX_REG)
    ti = {i: k for k, i in enumerate(tids)}
    c = c01[c01.image_id.isin(ti)]
    tr = c[(c.split == "train") & (c.category.isin(C.TB_CATS + ("healthy",)))]
    dv = c[c.split == "development"]
    ytr = np.array([1 if x in C.TB_CATS else 0 for x in tr.category])
    ydv = np.array([1 if x in C.TB_CATS else 0 for x in dv.category])
    log("TBX11K train %d (%d TB) / development %d (%d TB)"
        % (len(ytr), ytr.sum(), len(ydv), ydv.sum()))

    tr_idx = [ti[i] for i in tr.image_id]
    dv_idx = [ti[i] for i in dv.image_id]
    want = sorted(set(tr_idx) | set(dv_idx))
    X, kept = C.stream_regions(TBX_REG, keep_idx=want, device=dev)
    row = {int(g): k for k, g in enumerate(kept)}
    Rtr = X[[row[i] for i in tr_idx]]
    Rdv = X[[row[i] for i in dv_idx]]
    del X
    torch.cuda.empty_cache()

    with np.load(TBX_GLOBAL) as z:
        gid, TXG = z["image_id"].tolist(), z["emb"].astype(np.float32)
    gi = {i: k for k, i in enumerate(gid)}
    Gtr = TXG[[gi[i] for i in tr.image_id]]
    Gdv = TXG[[gi[i] for i in dv.image_id]]
    del TXG

    # ---------------- fit arm A (global) ----------------
    res, dauc_a, _ = H.train_seeds(Gtr, ytr, Gdv, ydv,
                                   H.TrainConfig(head_type="binary",
                                                 imbalance="logit_adjusted", max_epochs=60),
                                   seeds=SEEDS)
    a_heads = [(r.state_dict["weight"].numpy().astype(np.float32),
                r.state_dict["bias"].numpy().astype(np.float32)) for r in res]

    def global_score(G, heads):
        return np.mean([R.p_tb_from_logits(R.head_logits(G, W, b), 2) for W, b in heads],
                       axis=0)

    dev_a_ens = global_score(Gdv, a_heads)
    dev_a_s17 = global_score(Gdv, a_heads[:1])
    from sklearn.metrics import roc_auc_score
    log("arm A global: TBX11K dev AUROC  5-seed %.4f | seed17 %.4f | p2_36 mean %.4f "
        "[published %.4f]"
        % (roc_auc_score(ydv, dev_a_ens), roc_auc_score(ydv, dev_a_s17), dauc_a,
           EXPECT_DEV["A_global_pooler"]))

    # ---------------- fit arm B (regional max 12x12) ----------------
    b_heads = []
    for sd in SEEDS:
        W, b, best = C.train_head(Rtr, ytr, Rdv, ydv, how="max", seed=sd)
        b_heads.append((W, b))
        log("  arm B seed %2d  best dev AUROC %.4f" % (sd, best))
    dev_b = np.mean([C.score_head(Rdv, W, b, "max") for W, b in b_heads], axis=0)
    auc_b_dev = roc_auc_score(ydv, dev_b)
    log("arm B regional: TBX11K dev AUROC %.4f [published %.4f]"
        % (auc_b_dev, EXPECT_DEV["B_regional_max_12x12"]))

    ok = (abs(auc_b_dev - EXPECT_DEV["B_regional_max_12x12"]) < 0.01
          and abs(dauc_a - EXPECT_DEV["A_global_pooler"]) < 0.01)
    log("END-TO-END EQUIVALENCE AGAINST p2_36: %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit("arms do not reproduce the published TBX11K development AUCs")

    # ---------------- thresholds, from TBX11K development only ----------------
    thresholds = {
        "A_primary_5seed": float(np.quantile(dev_a_ens[ydv == 0], TARGET_SPEC)),
        "A_secondary_seed17": float(np.quantile(dev_a_s17[ydv == 0], TARGET_SPEC)),
        "B_regional_max_12x12": float(np.quantile(dev_b[ydv == 0], TARGET_SPEC)),
    }
    log("thresholds at %.0f%% specificity on TBX11K development: %s"
        % (100 * TARGET_SPEC, {k: round(v, 5) for k, v in thresholds.items()}))

    del Rtr, Rdv
    torch.cuda.empty_cache()

    # ---------------- DA/DB features (no labels yet) ----------------
    man = dadb_manifest()
    ids_order = [m["case_id"] for m in man]
    dr_ids = C.read_ids(DADB_REG)
    if dr_ids != ids_order:
        raise SystemExit("DA/DB regional id order does not match the source manifest")
    Dreg, _ = C.stream_regions(DADB_REG, device=dev)
    with np.load(DADB_GLOBAL) as z:
        Dglob = z["emb"].astype(np.float32)
    if Dglob.shape[0] != len(man):
        raise SystemExit("DA/DB global embedding has %d rows, expected %d"
                         % (Dglob.shape[0], len(man)))
    log("DA/DB features: regional %s (ids verified) | global %s (positional, "
        "order proven identical to the source manifest)" % (tuple(Dreg.shape), Dglob.shape))

    # ---------------- THE SINGLE SCORING PASS ----------------
    scores = {
        "A_primary_5seed": global_score(Dglob, a_heads),
        "A_secondary_seed17": global_score(Dglob, a_heads[:1]),
        "B_regional_max_12x12": np.mean(
            [C.score_head(Dreg, W, b, "max") for W, b in b_heads], axis=0),
    }
    del Dreg
    torch.cuda.empty_cache()

    # ---------------- write the manifest BEFORE any DA/DB metric ----------------
    manifest = {
        "run_id": RUN.name,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip(),
        "git_status_short": subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                           capture_output=True, text=True).stdout.split("\n"),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None},
        "backbone": {"model": "microsoft/rad-dino ViT-B/14, frozen",
                     "revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
                     "input_px": 518, "patch_grid": 37, "retrained": False},
        "configuration": {"preprocessing_variant": "lungs_only",
                          "region_grid": 12, "regions": 144,
                          "aggregation_arm_B": "max over 144 region logits",
                          "seeds": list(SEEDS),
                          "training_data": "TBX11K train, healthy-only negatives",
                          "n_train": int(len(ytr)), "n_train_tb": int(ytr.sum()),
                          "early_stop_on": "TBX11K development",
                          "n_dev": int(len(ydv)), "n_dev_tb": int(ydv.sum())},
        "arms": {"A_global_pooler": "RAD-DINO pooler_output, logit-adjusted binary head",
                 "B_regional_max_12x12": "12x12 region logits, max-pooled",
                 "primary": "both arms 5-seed ensemble (matched seed treatment)",
                 "secondary": "arm A seed 17 only, replicating p2_36 for comparability"},
        "thresholds_from_tbx11k_development_only": thresholds,
        "target_specificity": TARGET_SPEC,
        "tbx11k_dev_auroc": {"A_5seed": float(roc_auc_score(ydv, dev_a_ens)),
                             "A_seed17": float(roc_auc_score(ydv, dev_a_s17)),
                             "A_p2_36_mean": float(dauc_a),
                             "B_regional": float(auc_b_dev),
                             "published_reference": EXPECT_DEV,
                             "equivalence_passed": bool(ok)},
        "streaming_equivalence": equiv,
        "inclusion_rule": {
            "primary": "all 176 cases -- the committed pipeline scored all 176; the "
                       "segmentation sanity counter was incremented but no case excluded",
            "sensitivity": "175 cases, excluding %s" % SANITY_FAIL_ID,
            "exclusion_basis": "pre-existing label-blind validity rule "
                               "p1_build_c04_c05.roi_from_mask -> sanity_pass; %s fails on "
                               "'components' (1 connected lung field >=1%% of frame, "
                               "expected 2). Identified from mask geometry alone."
                               % SANITY_FAIL_ID,
            "label_blind": True},
        "no_tuning_guarantee": {
            "dadb_labels_used_to_fit_any_head": False,
            "dadb_labels_used_to_set_any_threshold": False,
            "dadb_used_to_select_seed_preprocessing_or_checkpoint": False,
            "offset_correction_applied": False,
            "rad_dino_retrained_or_finetuned": False,
            "pooling_sweep_rerun": False},
        "metrics_prespecified": ["AUROC + 95% bootstrap CI", "AUPRC with prevalence",
                                 "paired AUROC difference B-A with paired bootstrap CI",
                                 "sensitivity/specificity/confusion/referral at the "
                                 "TBX11K-development threshold"],
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "paired": True,
                      "same_resampled_cases_for_both_arms": True},
        "files": {str(p.relative_to(ROOT)): sha(p) for p in
                  (TBX_REG, TBX_GLOBAL, DADB_REG, DADB_GLOBAL,
                   ROOT / "artifacts/p2/dryrun/provisional_c01.csv",
                   HERE / "common.py", HERE / "score_dadb.py",
                   HERE / "extract_dadb_regional.py")},
        "cohort_manifest_sha256": hashlib.sha256(
            json.dumps(man, sort_keys=True).encode()).hexdigest(),
    }
    json.dump(manifest, io.open(RUN / "tables/experiment_manifest.json", "w",
                                encoding="utf-8"), indent=2)
    log("wrote tables/experiment_manifest.json (frozen before any DA/DB metric)")

    # ---------------- per-case predictions ----------------
    with io.open(RUN / "tables/predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "site", "label", "arm", "score",
                    "included_primary", "included_sensitivity"])
        for arm, s in scores.items():
            for k, m in enumerate(man):
                w.writerow([m["case_id"], m["site"], m["label"], arm, "%.10g" % s[k],
                            1, 0 if m["case_id"] == SANITY_FAIL_ID else 1])
    log("wrote tables/predictions.csv (%d rows)" % (len(man) * len(scores)))

    np.savez(RUN / "tables/heads.npz",
             **{("A_W_seed%d" % sd): a_heads[i][0] for i, sd in enumerate(SEEDS)},
             **{("A_b_seed%d" % sd): a_heads[i][1] for i, sd in enumerate(SEEDS)},
             **{("B_W_seed%d" % sd): b_heads[i][0].cpu().numpy() for i, sd in enumerate(SEEDS)},
             **{("B_b_seed%d" % sd): b_heads[i][1].cpu().numpy() for i, sd in enumerate(SEEDS)})
    log("wrote tables/heads.npz")


if __name__ == "__main__":
    main()
