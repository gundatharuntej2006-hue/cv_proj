# -*- coding: utf-8 -*-
"""Phase 2: the preregistered task-definition experiment.

Four arms differing ONLY in which TBX11K rows are positive and which are negative.
Protocol and acceptance criteria are fixed in 02_EXPERIMENT_PREREGISTRATION.md
(sha256 f748b4fc1cad44e6...), hashed before this script was run.

Nothing outside the run directory is written. No Vietnam label is used to fit anything.
TBX11K internal_test is not read.
"""
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import common as C                                                   # noqa: E402

TB_ALL = ("active_tb", "latent_tb", "active_latent_tb")
TB_ACTIVE = ("active_tb", "active_latent_tb")
ARMS = {
    "H_baseline_current":  {"pos": TB_ALL,    "neg": ("healthy",)},
    "L_latent_negative":   {"pos": TB_ACTIVE, "neg": ("healthy", "latent_tb")},
    "S_sick_negatives":    {"pos": TB_ALL,    "neg": ("healthy", "sick_non_tb")},
    "X_both_changes":      {"pos": TB_ACTIVE, "neg": ("healthy", "latent_tb", "sick_non_tb")},
}
AGG = "max"
N_BOOT = 4000
BOOT_SEED = 0
TARGET_SPEC = 0.70
MIN_EFFECT = 0.05
T0 = time.perf_counter()


def log(m):
    print("[%6.1fs] %s" % (time.perf_counter() - T0, m), flush=True)


def boot_ci(y, s, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        v.append(roc_auc_score(y[k], s[k]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired(y, pa, pb, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        d.append(roc_auc_score(y[k], pb[k]) - roc_auc_score(y[k], pa[k]))
    d = np.array(d)
    return {"delta": float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
            "ci95_lo": float(np.percentile(d, 2.5)),
            "ci95_hi": float(np.percentile(d, 97.5)),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0
                                  or np.percentile(d, 97.5) < 0),
            "n_resamples": int(len(d))}


def main():
    import torch
    import pandas as pd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device %s" % dev)

    # ---------------- TBX11K features: every train row plus development ----------------
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    tbx = C.REG / "regional_tbx11k_lungs_g12_v1.npz"
    tids = C.read_ids(tbx)
    ti = {i: k for k, i in enumerate(tids)}
    c = c01[c01.image_id.isin(ti)]
    tr_all = c[c.split == "train"]
    dv_all = c[c.split == "development"]
    assert "internal_test" not in set(tr_all.split) | set(dv_all.split)
    log("TBX11K rows available: train %d, development %d" % (len(tr_all), len(dv_all)))

    want = sorted({ti[i] for i in tr_all.image_id} | {ti[i] for i in dv_all.image_id})
    X, kept = C.stream_regions(tbx, keep_idx=want, device=dev)
    row = {int(g): k for k, g in enumerate(kept)}
    log("VRAM allocated %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    # ---------------- Vietnam features ----------------
    vn = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    vids = C.read_ids(vn)
    V, _ = C.stream_regions(vn, device=dev)
    log("VRAM allocated %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    meta = {str(r["pid"]): r for r in csv.DictReader(io.open(
        ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv", encoding="utf-8-sig"))}
    y_vn = np.array([int(meta[p]["xpert_result"]) for p in vids])
    past = np.array([int(meta[p]["past_tb"]) for p in vids])
    lunit = np.array([float(meta[p]["lunit_3901"]) for p in vids])
    split = json.load(io.open(ROOT / "artifacts/p2/external/vietnam_tune_report_split.json",
                              encoding="utf-8"))
    tune = set(str(x) for x in split["tune"])
    is_tune = np.array([p in tune for p in vids])

    # ---------------- fit every arm ----------------
    results, scores, devinfo = {}, {}, {}
    for name, spec in ARMS.items():
        tr = tr_all[tr_all.category.isin(spec["pos"] + spec["neg"])]
        dv = dv_all[dv_all.category.isin(spec["pos"] + spec["neg"])]
        ytr = np.array([1 if x in spec["pos"] else 0 for x in tr.category])
        ydv = np.array([1 if x in spec["pos"] else 0 for x in dv.category])
        Xtr = X[[row[ti[i]] for i in tr.image_id]]
        Xdv = X[[row[ti[i]] for i in dv.image_id]]
        heads = []
        for sd in C.SEEDS:
            W, b, best = C.train_head(Xtr, ytr, Xdv, ydv, how=AGG, seed=sd)
            heads.append((W, b))
        dv_s = np.mean([C.score_head(Xdv, W, b, AGG) for W, b in heads], axis=0)
        dev_auc = roc_auc_score(ydv, dv_s)
        thr = float(np.quantile(dv_s[ydv == 0], TARGET_SPEC))
        devinfo[name] = {"n_train": int(len(ytr)), "n_train_pos": int(ytr.sum()),
                         "n_dev": int(len(ydv)), "n_dev_pos": int(ydv.sum()),
                         "tbx11k_dev_auroc": float(dev_auc),
                         "threshold_at_%d_spec" % int(TARGET_SPEC * 100): thr,
                         "positives": list(spec["pos"]), "negatives": list(spec["neg"])}
        scores[name] = np.mean([C.score_head(V, W, b, AGG) for W, b in heads], axis=0)
        log("arm %-20s train %-5d (%3d pos)  TBX11K dev AUROC %.4f  thr %.4f"
            % (name, len(ytr), ytr.sum(), dev_auc, thr))
        del Xtr, Xdv
        torch.cuda.empty_cache()

    del X, V
    torch.cuda.empty_cache()

    # sanity: the baseline arm must reproduce the published Vietnam figure
    base_full = roc_auc_score(y_vn, scores["H_baseline_current"])
    log("arm H Vietnam full-cohort AUROC %.4f  [published 0.7240]" % base_full)
    if abs(base_full - 0.7240) > 0.01:
        raise SystemExit("baseline arm does not reproduce the published Vietnam figure")

    # ---------------- endpoints ----------------
    tasks = {
        "PRIMARY active_TB_vs_pastTB_negatives": (y_vn == 1) | ((y_vn == 0) & (past == 1)),
        "active_TB_vs_noPastTB_negatives": (y_vn == 1) | ((y_vn == 0) & (past == 0)),
        "GUARDRAIL active_TB_vs_all_negatives": np.ones(len(y_vn), bool),
    }
    out = {"arms": devinfo, "endpoints": {}, "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED}}

    for tname, mask in tasks.items():
        yy = y_vn[mask]
        blk = {"n": int(mask.sum()), "n_pos": int(yy.sum()),
               "n_neg": int((yy == 0).sum()),
               "lunit_auroc": float(roc_auc_score(yy, lunit[mask])), "arms": {}}
        for name in ARMS:
            ss = scores[name][mask]
            lo, hi = boot_ci(yy, ss)
            blk["arms"][name] = {
                "auroc": float(roc_auc_score(yy, ss)), "ci95": [lo, hi],
                "auprc": float(average_precision_score(yy, ss)),
                "vs_H": paired(yy, scores["H_baseline_current"][mask], ss)
                        if name != "H_baseline_current" else None}
        out["endpoints"][tname] = blk
        print("\n%s  (n=%d, %d TB, %d neg)  Lunit %.4f"
              % (tname, blk["n"], blk["n_pos"], blk["n_neg"], blk["lunit_auroc"]))
        for name in ARMS:
            a = blk["arms"][name]
            v = a["vs_H"]
            print("  %-20s AUROC %.4f [%.4f, %.4f]%s"
                  % (name, a["auroc"], a["ci95"][0], a["ci95"][1],
                     "" if v is None else "   vs H %+0.4f [%+0.4f, %+0.4f] %s"
                     % (v["delta"], v["ci95_lo"], v["ci95_hi"],
                        "EXCLUDES 0" if v["excludes_zero"] else "includes 0")))

    # by half, for transparency about exposure
    out["by_half"] = {}
    for half, hm in (("TUNE", is_tune), ("REPORT", ~is_tune)):
        m = hm & ((y_vn == 1) | ((y_vn == 0) & (past == 1)))
        out["by_half"][half] = {
            "n": int(m.sum()), "n_pos": int(y_vn[m].sum()),
            **{name: round(float(roc_auc_score(y_vn[m], scores[name][m])), 4)
               for name in ARMS}}
    print("\nprimary endpoint by half (both exposed; reported for transparency):")
    for h, v in out["by_half"].items():
        print("  %-7s n=%-5d  " % (h, v["n"]) + "  ".join(
            "%s %.4f" % (k.split("_")[0], v[k]) for k in ARMS))

    # ---------------- the decision rule, applied as written ----------------
    p = out["endpoints"]["PRIMARY active_TB_vs_pastTB_negatives"]
    g = out["endpoints"]["GUARDRAIL active_TB_vs_all_negatives"]
    dx = p["arms"]["X_both_changes"]["vs_H"]
    c1 = bool(dx["delta"] >= MIN_EFFECT)
    c2 = bool(dx["excludes_zero"])
    drop = g["arms"]["H_baseline_current"]["auroc"] - g["arms"]["X_both_changes"]["auroc"]
    c3 = bool(drop <= 0.02)
    verdict = "ACCEPT" if (c1 and c2 and c3) else (
        "INCONCLUSIVE" if (not c2 and abs(dx["delta"]) < MIN_EFFECT) else "REJECT")
    out["decision"] = {
        "minimum_effect_size": MIN_EFFECT,
        "criterion_1_effect_at_least_min": c1, "observed_delta": dx["delta"],
        "criterion_2_ci_excludes_zero": c2, "ci95": [dx["ci95_lo"], dx["ci95_hi"]],
        "criterion_3_guardrail_full_cohort_drop_le_0.02": c3,
        "full_cohort_drop": float(drop),
        "VERDICT": verdict,
        "prereg_sha256": (RUN / "tables/prereg_hash.txt").read_text(
            encoding="utf-8").split()[0],
    }
    print("\n=== PREREGISTERED DECISION ===")
    print(json.dumps(out["decision"], indent=2))

    # ---------------- per-case predictions, run directory only ----------------
    with io.open(RUN / "tables/phase2_predictions.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "xpert_result", "past_tb", "split_half", "lunit_3901"]
                   + ["score_" + a for a in ARMS])
        for k, p_ in enumerate(vids):
            w.writerow([p_, int(y_vn[k]), int(past[k]),
                        "TUNE" if is_tune[k] else "REPORT", "%.6f" % lunit[k]]
                       + ["%.6f" % scores[a][k] for a in ARMS])
    json.dump(out, io.open(RUN / "tables/phase2_metrics.json", "w", encoding="utf-8"),
              indent=2)
    log("wrote tables/phase2_predictions.csv and tables/phase2_metrics.json")


if __name__ == "__main__":
    main()
