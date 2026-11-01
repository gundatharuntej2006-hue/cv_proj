# -*- coding: utf-8 -*-
"""Phase 2 + 3: metrics from the frozen per-case predictions, and integrity checks.

Reads tables/predictions.csv only. Fits nothing, tunes nothing, and cannot alter any arm.
Every AUROC and AUPRC is computed with sklearn.metrics, independently of the project's
own selection.auc.
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

N_BOOT = 4000
BOOT_SEED = 0
SANITY_FAIL_ID = "px4"
ARM_A = "A_primary_5seed"
ARM_A2 = "A_secondary_seed17"
ARM_B = "B_regional_max_12x12"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def boot_auc(y, p, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        v.append(roc_auc_score(y[k], p[k]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired(y, pa, pb, seed=BOOT_SEED, keep=False):
    """AUROC(pb) - AUROC(pa) on identical resampled cases."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        d.append(roc_auc_score(y[k], pb[k]) - roc_auc_score(y[k], pa[k]))
    d = np.array(d)
    out = {"delta": float(roc_auc_score(y, pb) - roc_auc_score(y, pa)),
           "ci95_lo": float(np.percentile(d, 2.5)),
           "ci95_hi": float(np.percentile(d, 97.5)),
           "n_resamples_used": int(len(d)),
           "frac_favouring_B": float((d > 0).mean()),
           "median": float(np.median(d)), "sd": float(d.std(ddof=1))}
    out["ci_excludes_zero"] = bool(out["ci95_lo"] > 0 or out["ci95_hi"] < 0)
    return (out, d) if keep else (out, None)


def op(y, s, thr):
    f = s >= thr
    tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum()); tn = int((~f & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    sl, sh = wilson(tp, tp + fn)
    pl, ph = wilson(tn, tn + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "sens_ci95": [sl, sh],
            "specificity": spec, "spec_ci95": [pl, ph],
            "ppv": tp / max(tp + fp, 1) if tp + fp else None,
            "npv": tn / max(tn + fn, 1) if tn + fn else None,
            "referral_rate": (tp + fp) / len(y)}


def main():
    man = json.load(io.open(RUN / "tables/experiment_manifest.json", encoding="utf-8"))
    thr = man["thresholds_from_tbx11k_development_only"]

    rows = list(csv.DictReader(io.open(RUN / "tables/predictions.csv", encoding="utf-8-sig")))
    arms = sorted({r["arm"] for r in rows})
    by = {a: [r for r in rows if r["arm"] == a] for a in arms}
    base_ids = [r["case_id"] for r in by[arms[0]]]
    for a in arms:
        assert [r["case_id"] for r in by[a]] == base_ids, "arm %s case order differs" % a

    ids = np.array(base_ids)
    y = np.array([int(r["label"]) for r in by[arms[0]]])
    site = np.array([r["site"] for r in by[arms[0]]])
    S = {a: np.array([float(r["score"]) for r in by[a]]) for a in arms}

    # ------------------------------------------------ integrity
    integ = {
        "n_cases": int(len(ids)),
        "n_unique_case_ids": int(len(set(ids.tolist()))),
        "duplicates": int(len(ids) - len(set(ids.tolist()))),
        "arms": arms,
        "all_arms_same_case_order": True,
        "non_finite_scores": {a: int((~np.isfinite(S[a])).sum()) for a in arms},
        "n_unique_scores": {a: int(len(np.unique(S[a]))) for a in arms},
        "score_ranges": {a: [float(S[a].min()), float(S[a].max())] for a in arms},
        "arm_arrays_identical": {
            "%s|%s" % (a, b): bool(np.array_equal(S[a], S[b]))
            for i, a in enumerate(arms) for b in arms[i + 1:]},
        "labels_rederived_from_filename_match": None,
        "by_site": {s: {"n": int((site == s).sum()), "pos": int(y[site == s].sum()),
                        "neg": int((y[site == s] == 0).sum())} for s in ("DA", "DB")},
        "pooled": {"n": int(len(y)), "pos": int(y.sum()), "neg": int((y == 0).sum()),
                   "prevalence": float(y.mean())},
    }
    # re-derive labels straight from the filesystem, independent of predictions.csv
    SRC = ROOT / "data/raw/tbx11k/TBX11K/imgs/extra/da+db"
    truth = {}
    for p in sorted(SRC.rglob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            truth[p.stem] = (1 if p.name[0] == "p" else 0,
                             "DA" if p.suffix.lower() == ".png" else "DB")
    integ["labels_rederived_from_filename_match"] = bool(
        all(truth[i][0] == yy for i, yy in zip(ids, y)))
    integ["sites_rederived_match"] = bool(all(truth[i][1] == ss for i, ss in zip(ids, site)))
    integ["published_counts_match"] = (
        integ["by_site"]["DA"] == {"n": 72, "pos": 22, "neg": 50}
        and integ["by_site"]["DB"] == {"n": 104, "pos": 52, "neg": 52})

    # ------------------------------------------------ metrics
    def block(mask, label, inclusion):
        yy = y[mask]
        out = {"group": label, "inclusion": inclusion, "n": int(mask.sum()),
               "n_pos": int(yy.sum()), "n_neg": int((yy == 0).sum()),
               "prevalence": float(yy.mean()), "arms": {}}
        for a in arms:
            ss = S[a][mask]
            lo, hi = boot_auc(yy, ss)
            t = thr[a]
            out["arms"][a] = {
                "auroc": float(roc_auc_score(yy, ss)),
                "auroc_ci95": [lo, hi],
                "auprc": float(average_precision_score(yy, ss)),
                "auprc_baseline_prevalence": float(yy.mean()),
                "threshold_from_tbx11k_dev": float(t),
                "operating_point": op(yy, ss, t)}
        d1, dist = paired(yy, S[ARM_A][mask], S[ARM_B][mask], keep=(label == "POOLED"))
        d2, _ = paired(yy, S[ARM_A2][mask], S[ARM_B][mask])
        out["paired_B_minus_A_primary"] = d1
        out["paired_B_minus_A_secondary_seed17"] = d2
        return out, dist

    results = {"primary_all_176": [], "sensitivity_175_excl_%s" % SANITY_FAIL_ID: []}
    keep_all = np.ones(len(y), bool)
    keep_175 = ids != SANITY_FAIL_ID

    pooled_dist = None
    for incl, mask_all in (("primary_all_176", keep_all),
                           ("sensitivity_175_excl_%s" % SANITY_FAIL_ID, keep_175)):
        for s in ("DA", "DB"):
            b, _ = block(mask_all & (site == s), s, incl)
            results[incl].append(b)
        b, dist = block(mask_all, "POOLED", incl)
        results[incl].append(b)
        if incl == "primary_all_176":
            pooled_dist = dist

    # bootstrap distribution for reproducibility of the headline interval
    if pooled_dist is not None:
        with io.open(RUN / "tables/paired_bootstrap.csv", "w", encoding="utf-8",
                     newline="") as f:
            w = csv.writer(f)
            w.writerow(["resample_index", "delta_auroc_B_minus_A_primary_pooled_all176"])
            for i, v in enumerate(pooled_dist):
                w.writerow([i, "%.10g" % v])

    # ------------------------------------------------ seed sensitivity (5 seeds were
    # pre-specified in the original protocol; this is not post-hoc seed hunting)
    import torch
    z = np.load(RUN / "tables/heads.npz")
    seeds = man["configuration"]["seeds"]
    sys.path.insert(0, str(HERE))
    import common as C
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D, _ = C.stream_regions(RUN / "tables/regional_dadb_lungs_g12_v1.npz", device=dev,
                            verbose=False)
    seed_rows = []
    for sd in seeds:
        W = torch.from_numpy(z["B_W_seed%d" % sd]).to(dev)
        b = torch.from_numpy(z["B_b_seed%d" % sd]).to(dev)
        s = C.score_head(D, W, b, "max")
        seed_rows.append({"seed": int(sd),
                          "auroc_pooled": float(roc_auc_score(y, s)),
                          "auroc_DA": float(roc_auc_score(y[site == "DA"], s[site == "DA"])),
                          "auroc_DB": float(roc_auc_score(y[site == "DB"], s[site == "DB"]))})
    del D
    torch.cuda.empty_cache()

    out = {"results": results, "integrity": integ,
           "seed_sensitivity_arm_B": seed_rows,
           "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
           "thresholds": thr,
           "manifest_sha256": hashlib.sha256(
               (RUN / "tables/experiment_manifest.json").read_bytes()).hexdigest()}
    json.dump(out, io.open(RUN / "tables/metrics.json", "w", encoding="utf-8"), indent=2)

    # ------------------------------------------------ console summary
    print("\n=== INTEGRITY ===")
    for k in ("n_cases", "duplicates", "labels_rederived_from_filename_match",
              "sites_rederived_match", "published_counts_match"):
        print("  %-38s %s" % (k, integ[k]))
    print("  non-finite scores:   %s" % integ["non_finite_scores"])
    print("  any two arms identical: %s" % integ["arm_arrays_identical"])

    for incl in results:
        print("\n=== %s ===" % incl.upper())
        for b in results[incl]:
            print("  %-7s n=%-4d pos=%-3d prev=%.3f" % (b["group"], b["n"], b["n_pos"],
                                                        b["prevalence"]))
            for a in arms:
                m = b["arms"][a]
                o = m["operating_point"]
                print("    %-20s AUROC %.4f [%.4f, %.4f]  AUPRC %.4f  sens %.3f spec %.3f "
                      "refer %.3f"
                      % (a, m["auroc"], m["auroc_ci95"][0], m["auroc_ci95"][1], m["auprc"],
                         o["sensitivity"], o["specificity"], o["referral_rate"]))
            d = b["paired_B_minus_A_primary"]
            print("    paired B-A (primary): %+0.4f [%+0.4f, %+0.4f]  %s"
                  % (d["delta"], d["ci95_lo"], d["ci95_hi"],
                     "EXCLUDES 0" if d["ci_excludes_zero"] else "includes 0"))

    print("\n=== SEED SENSITIVITY, arm B ===")
    for r in seed_rows:
        print("  seed %-3d pooled %.4f  DA %.4f  DB %.4f"
              % (r["seed"], r["auroc_pooled"], r["auroc_DA"], r["auroc_DB"]))


if __name__ == "__main__":
    main()
