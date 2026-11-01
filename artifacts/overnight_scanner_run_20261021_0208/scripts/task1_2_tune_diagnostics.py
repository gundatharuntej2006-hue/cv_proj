# -*- coding: utf-8 -*-
"""Tasks 1 and 2, TUNE only. No REPORT value is computed or printed by this script.

Task 1 recomputes per-scanner diagnostics from the committed case-level score file.
Task 2 designs the scanner-specific thresholds and measures their sampling uncertainty.

The score column is the focal-pooling score (12x12 regions, max aggregation, seeds
17/29/43/61/79, head trained on TBX11K train). Provenance read from
scripts/p2_35_gate_operational_audit.py lines 72-101, not inferred from the column name.
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import common as C                                                   # noqa: E402

TARGET_SPEC = 0.70
N_BOOT = 2000
SEED = 17


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def op_metrics(y, s, thr):
    flag = s >= thr
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1) if (tp + fp) else float("nan")
    npv = tn / max(tn + fn, 1) if (tn + fn) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sens=sens, spec=spec, ppv=ppv, npv=npv,
                referral_rate=(tp + fp) / len(y))


def main():
    rows = list(csv.DictReader(io.open(
        ROOT / "artifacts/p2/audit/vietnam_case_level_gate_audit.csv", encoding="utf-8-sig")))
    pid = np.array([r["pid"] for r in rows])
    y = np.array([int(r["xpert_result"]) for r in rows])
    s = np.array([float(r["classifier_score"]) for r in rows])
    scan = np.array([int(r["rad_system"]) for r in rows])
    half = np.array([r["split_half"] for r in rows])

    scanners = sorted(set(scan.tolist()))
    print("scanner values present in the manifest: %s  (n=%d)" % (scanners, len(scanners)))
    assert len(set(pid)) == len(pid), "duplicate pid"

    tune = half == "TUNE"
    print("TUNE n=%d  REPORT n=%d  (REPORT is NOT touched by this script)"
          % (tune.sum(), (half == "REPORT").sum()))

    # ---------------- Task 1: per-scanner diagnostics on TUNE ----------------
    pooled_thr = float(np.quantile(s[tune & (y == 0)], TARGET_SPEC))
    print("\npooled single threshold, %.0f%% specificity on TUNE non-TB: %.6f"
          % (100 * TARGET_SPEC, pooled_thr))

    t1 = []
    for g in scanners + ["ALL"]:
        m = tune if g == "ALL" else (tune & (scan == g))
        yy, ss = y[m], s[m]
        auc = roc_auc_score(yy, ss)
        lo, hi = C.boot_ci(yy, ss, n_boot=N_BOOT, seed=SEED)
        ap = average_precision_score(yy, ss)
        o = op_metrics(yy, ss, pooled_thr)
        sl, sh = wilson(o["tp"], o["tp"] + o["fn"])
        pl, ph = wilson(o["tn"], o["tn"] + o["fp"])
        t1.append([g, int(m.sum()), int(yy.sum()), int((yy == 0).sum()),
                   round(float(yy.mean()), 4), round(float(auc), 4), round(lo, 4), round(hi, 4),
                   round(float(ap), 4), round(pooled_thr, 6),
                   o["tp"], o["fp"], o["fn"], o["tn"],
                   round(o["sens"], 4), round(sl, 4), round(sh, 4),
                   round(o["spec"], 4), round(pl, 4), round(ph, 4),
                   round(o["ppv"], 4), round(o["npv"], 4), round(o["referral_rate"], 4)])

    hdr1 = ["group", "n", "n_tb", "n_non_tb", "prevalence", "auroc", "auroc_lo", "auroc_hi",
            "auprc", "pooled_threshold", "tp", "fp", "fn", "tn",
            "sensitivity", "sens_lo", "sens_hi", "specificity", "spec_lo", "spec_hi",
            "ppv", "npv", "referral_rate"]
    with io.open(RUN / "tables/baseline_by_scanner.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr1)
        w.writerows(t1)
    print("\nTask 1 - TUNE diagnostics at the single pooled threshold")
    print("  %-6s %6s %5s %7s %7s %7s %7s %7s" %
          ("group", "n", "tb", "prev", "AUROC", "sens", "spec", "refer"))
    for r in t1:
        print("  %-6s %6d %5d %7.3f %7.4f %7.3f %7.3f %7.3f"
              % (r[0], r[1], r[2], r[4], r[5], r[14], r[17], r[22]))

    # ---------------- Task 2: scanner-specific thresholds, designed on TUNE ----------------
    # Procedure, stated explicitly: for each scanner g, the threshold is the empirical
    # TARGET_SPEC quantile of the scores of TUNE cases with xpert_result == 0 and
    # rad_system == g, using numpy's default linear interpolation between order
    # statistics. No visual inspection, no REPORT involvement.
    local_thr = {}
    for g in scanners:
        m = tune & (scan == g) & (y == 0)
        local_thr[g] = float(np.quantile(s[m], TARGET_SPEC))

    t2 = []
    for g in scanners:
        m = tune & (scan == g)
        yy, ss = y[m], s[m]
        a = op_metrics(yy, ss, pooled_thr)
        b = op_metrics(yy, ss, local_thr[g])
        n_neg = int((yy == 0).sum())
        t2.append([g, int(m.sum()), int(yy.sum()), n_neg,
                   round(pooled_thr, 6), round(local_thr[g], 6),
                   round(local_thr[g] - pooled_thr, 6),
                   round(a["sens"], 4), round(b["sens"], 4),
                   round(a["spec"], 4), round(b["spec"], 4),
                   round(a["referral_rate"], 4), round(b["referral_rate"], 4),
                   round(a["ppv"], 4), round(b["ppv"], 4),
                   a["fn"], b["fn"], a["fp"], b["fp"]])
    hdr2 = ["scanner", "n_tune", "n_tb", "n_non_tb", "pooled_threshold", "scanner_threshold",
            "threshold_shift", "sens_pooled", "sens_scanner", "spec_pooled", "spec_scanner",
            "referral_pooled", "referral_scanner", "ppv_pooled", "ppv_scanner",
            "missed_tb_pooled", "missed_tb_scanner", "false_pos_pooled", "false_pos_scanner"]
    with io.open(RUN / "tables/threshold_calibration.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr2)
        w.writerows(t2)
    print("\nTask 2 - scanner thresholds designed on TUNE")
    print("  %-8s %10s %10s %8s %8s %8s %8s" %
          ("scanner", "pooled_t", "local_t", "spec_p", "spec_l", "refer_p", "refer_l"))
    for r in t2:
        print("  %-8d %10.4f %10.4f %8.3f %8.3f %8.3f %8.3f"
              % (r[0], r[4], r[5], r[9], r[10], r[11], r[12]))

    # ---------------- threshold uncertainty: stratified bootstrap on TUNE ----------------
    rng = np.random.default_rng(SEED)
    boot = {g: [] for g in scanners}
    boot["POOLED"] = []
    neg_idx = {g: np.where(tune & (scan == g) & (y == 0))[0] for g in scanners}
    all_neg = np.where(tune & (y == 0))[0]
    for _ in range(N_BOOT):
        for g in scanners:
            k = rng.choice(neg_idx[g], len(neg_idx[g]), replace=True)
            boot[g].append(float(np.quantile(s[k], TARGET_SPEC)))
        k = rng.choice(all_neg, len(all_neg), replace=True)
        boot["POOLED"].append(float(np.quantile(s[k], TARGET_SPEC)))

    t3 = []
    for g in scanners + ["POOLED"]:
        v = np.array(boot[g])
        n_neg = len(all_neg) if g == "POOLED" else len(neg_idx[g])
        point = pooled_thr if g == "POOLED" else local_thr[g]
        width = float(np.percentile(v, 97.5) - np.percentile(v, 2.5))
        t3.append([g, n_neg, round(point, 6), round(float(v.mean()), 6),
                   round(float(v.std(ddof=1)), 6),
                   round(float(np.percentile(v, 2.5)), 6),
                   round(float(np.percentile(v, 97.5)), 6), round(width, 6),
                   "UNSTABLE" if width > 1.0 else "ok",
                   "SMALL SUBGROUP" if n_neg < 300 else ""])
    with io.open(RUN / "tables/threshold_bootstrap.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_non_tb_tune", "threshold", "boot_mean", "boot_sd",
                    "ci95_lo", "ci95_hi", "ci95_width", "stability_flag", "size_flag"])
        w.writerows(t3)
    print("\n  threshold uncertainty (stratified bootstrap on TUNE non-TB, %d resamples)"
          % N_BOOT)
    for r in t3:
        print("    %-8s n_neg=%-5d t=%8.4f  CI [%7.4f, %7.4f] width %.4f  %s %s"
              % (r[0], r[1], r[2], r[5], r[6], r[7], r[8], r[9]))

    json.dump({"target_specificity": TARGET_SPEC, "pooled_threshold": pooled_thr,
               "scanner_thresholds": {str(k): v for k, v in local_thr.items()},
               "scanners": scanners, "n_boot": N_BOOT, "seed": SEED,
               "quantile_method": "numpy.quantile, linear interpolation, on TUNE non-TB scores",
               "report_touched": False},
              io.open(RUN / "tables/thresholds_tune.json", "w", encoding="utf-8"), indent=2)
    print("\nTUNE-only phase complete. REPORT was not read.")


if __name__ == "__main__":
    main()
