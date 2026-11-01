# -*- coding: utf-8 -*-
"""Task 5: is the scanner disparity an offset, or is it a ranking failure? TUNE only.

Every calibrator and threshold here is fitted on TUNE. No REPORT value is computed.
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import common as C                                                   # noqa: E402

TARGET_SPEC = 0.70
SEED = 17


def platt(s, y):
    """1-D logistic calibration. Returns (slope a, intercept b) with p = sigmoid(a*s + b)."""
    lr = LogisticRegression(C=1e6, max_iter=5000, random_state=SEED)
    lr.fit(s.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cal_slope_intercept(p, y):
    """Cox calibration: regress y on logit(p). Perfect calibration is slope 1, intercept 0."""
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(C=1e6, max_iter=5000, random_state=SEED)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def main():
    rows = list(csv.DictReader(io.open(
        RUN / "tables/vietnam_case_scores.csv", encoding="utf-8-sig")))
    y = np.array([int(r["xpert_result"]) for r in rows])
    s = np.array([float(r["score_baseline"]) for r in rows])
    scan = np.array([int(r["rad_system"]) for r in rows])
    half = np.array([r["split_half"] for r in rows])
    tune = half == "TUNE"
    scanners = sorted(set(scan.tolist()))

    # ---------- 1. ranking vs offset ----------
    out = []
    pooled_a, pooled_b = platt(s[tune], y[tune])
    for g in scanners:
        m = tune & (scan == g)
        yy, ss = y[m], s[m]
        auc = roc_auc_score(yy, ss)
        lo, hi = C.boot_ci(yy, ss, n_boot=2000, seed=SEED)
        # pooled calibrator applied to this scanner
        p_pool = sigmoid(pooled_a * ss + pooled_b)
        sl_p, ic_p = cal_slope_intercept(p_pool, yy)
        br_p = brier_score_loss(yy, p_pool)
        # scanner-specific calibrator, fitted on this scanner's TUNE rows
        a_g, b_g = platt(ss, yy)
        p_loc = sigmoid(a_g * ss + b_g)
        sl_l, ic_l = cal_slope_intercept(p_loc, yy)
        br_l = brier_score_loss(yy, p_loc)
        out.append([g, int(m.sum()), int(yy.sum()), round(float(yy.mean()), 4),
                    round(float(auc), 4), round(lo, 4), round(hi, 4),
                    round(float(np.median(ss[yy == 0])), 4),
                    round(float(np.median(ss[yy == 1])), 4),
                    round(float(ss[yy == 0].std(ddof=1)), 4),
                    round(pooled_a, 4), round(pooled_b, 4),
                    round(br_p, 4), round(sl_p, 4), round(ic_p, 4),
                    round(a_g, 4), round(b_g, 4),
                    round(br_l, 4), round(sl_l, 4), round(ic_l, 4)])

    with io.open(RUN / "tables/calibration_by_scanner.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["scanner", "n_tune", "n_tb", "prevalence", "auroc", "auroc_lo", "auroc_hi",
                    "median_score_non_tb", "median_score_tb", "sd_score_non_tb",
                    "pooled_platt_a", "pooled_platt_b", "brier_pooled",
                    "cal_slope_pooled", "cal_intercept_pooled",
                    "scanner_platt_a", "scanner_platt_b", "brier_scanner",
                    "cal_slope_scanner", "cal_intercept_scanner"])
        w.writerows(out)

    print("Task 5 - ranking versus offset, TUNE only")
    print("  %-4s %6s %7s %9s %9s %9s %9s %9s" %
          ("scan", "n", "AUROC", "med_neg", "med_pos", "brier_p", "brier_l", "slope_p"))
    for r in out:
        print("  %-4d %6d %7.4f %9.3f %9.3f %9.4f %9.4f %9.3f"
              % (r[0], r[1], r[4], r[7], r[8], r[12], r[17], r[13]))

    # ---------- 2. separation statistic: how much is pure shift? ----------
    # If the disparity were pure offset, subtracting each scanner's non-TB median would
    # align the distributions and leave AUROC unchanged (AUROC is shift-invariant within
    # a group). What changes is the POOLED behaviour at one threshold.
    shifted = s.copy()
    for g in scanners:
        m = scan == g
        shifted[m] = s[m] - np.median(s[tune & m & (y == 0)])
    pooled_before = roc_auc_score(y[tune], s[tune])
    pooled_after = roc_auc_score(y[tune], shifted[tune])
    thr_before = float(np.quantile(s[tune & (y == 0)], TARGET_SPEC))
    thr_after = float(np.quantile(shifted[tune & (y == 0)], TARGET_SPEC))

    rr = []
    for g in scanners:
        m = tune & (scan == g)
        rb = float((s[m] >= thr_before).mean())
        ra = float((shifted[m] >= thr_after).mean())
        sb = float((s[m & (y == 1)] >= thr_before).mean())
        sa = float((shifted[m & (y == 1)] >= thr_after).mean())
        rr.append([g, round(rb, 4), round(ra, 4), round(sb, 4), round(sa, 4)])

    print("\n  pooled AUROC before median-shift alignment: %.4f" % pooled_before)
    print("  pooled AUROC after  median-shift alignment: %.4f  (change %+0.4f)"
          % (pooled_after, pooled_after - pooled_before))
    print("  referral rate by scanner, before -> after alignment:")
    for g, rb, ra, sb, sa in rr:
        print("    scanner %d: %.3f -> %.3f   (sensitivity %.3f -> %.3f)" % (g, rb, ra, sb, sa))

    spread_before = max(x[1] for x in rr) - min(x[1] for x in rr)
    spread_after = max(x[2] for x in rr) - min(x[2] for x in rr)
    print("  referral-rate spread across scanners: %.3f -> %.3f" % (spread_before, spread_after))

    json.dump({"target_specificity": TARGET_SPEC,
               "pooled_auroc_before_alignment": float(pooled_before),
               "pooled_auroc_after_alignment": float(pooled_after),
               "threshold_before": thr_before, "threshold_after": thr_after,
               "referral_by_scanner": {str(r[0]): {"before": r[1], "after": r[2],
                                                   "sens_before": r[3], "sens_after": r[4]}
                                       for r in rr},
               "referral_spread_before": round(spread_before, 4),
               "referral_spread_after": round(spread_after, 4),
               "alignment_statistic": "per-scanner subtraction of the TUNE non-TB median",
               "note": "AUROC within a scanner is shift-invariant, so any pooled AUROC "
                       "change here comes only from re-ordering cases ACROSS scanners",
               "report_touched": False},
              io.open(RUN / "tables/offset_vs_ranking.json", "w", encoding="utf-8"), indent=2)
    print("\nTUNE-only. REPORT untouched.")


if __name__ == "__main__":
    main()
