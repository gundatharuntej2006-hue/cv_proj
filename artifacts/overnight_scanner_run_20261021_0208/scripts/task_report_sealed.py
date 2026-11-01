# -*- coding: utf-8 -*-
"""The single sealed REPORT pass. Runs only after FROZEN_MANIFEST.md exists.

Reads every parameter from tables/frozen_parameters.json. Fits nothing. Selects nothing.
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
sys.path.insert(0, str(HERE))
import common as C                                                   # noqa: E402

N_BOOT = 4000
SEED = 0


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def op(y, s, thr):
    f = s >= thr
    tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum()); tn = int((~f & (y == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1) if tp + fp else float("nan"),
                npv=tn / max(tn + fn, 1) if tn + fn else float("nan"),
                referral=(tp + fp) / len(y))


def main():
    assert (RUN / "FROZEN_MANIFEST.md").exists(), "manifest not frozen; refusing to score"
    P = json.load(io.open(RUN / "tables/frozen_parameters.json", encoding="utf-8"))

    rows = list(csv.DictReader(io.open(
        RUN / "tables/vietnam_case_scores.csv", encoding="utf-8-sig")))
    y = np.array([int(r["xpert_result"]) for r in rows])
    scan = np.array([int(r["rad_system"]) for r in rows])
    half = np.array([r["split_half"] for r in rows])
    S = {k: np.array([float(r[k]) for r in rows]) for k in
         ("score_baseline", "score_P1_projected_frozen_head",
          "score_P2_projected_retrained_head")}
    rep = half == "REPORT"
    scanners = sorted(set(scan.tolist()))
    print("sealed REPORT pass: n=%d, TB=%d" % (rep.sum(), y[rep].sum()))

    # offset-corrected score, using TUNE-derived offsets only
    off = P["scanner_offsets_tune_non_tb_median"]
    S["score_offset_corrected"] = S["score_baseline"].copy()
    for g in scanners:
        S["score_offset_corrected"][scan == g] -= off[str(g)]

    # ---------------- A. discrimination ----------------
    disc = []
    base = S["score_baseline"][rep]
    for name in ("score_baseline", "score_P1_projected_frozen_head",
                 "score_P2_projected_retrained_head", "score_offset_corrected"):
        s = S[name][rep]
        for g in ["POOLED"] + scanners:
            m = np.ones(rep.sum(), bool) if g == "POOLED" else (scan[rep] == g)
            yy, ss = y[rep][m], s[m]
            auc = roc_auc_score(yy, ss)
            lo, hi = C.boot_ci(yy, ss, n_boot=N_BOOT, seed=SEED)
            ap = average_precision_score(yy, ss)
            d, dlo, dhi = (0.0, 0.0, 0.0) if name == "score_baseline" else \
                C.paired_boot(yy, base[m], ss, n_boot=N_BOOT, seed=SEED)
            disc.append([name, str(g), int(m.sum()), int(yy.sum()),
                         round(float(yy.mean()), 4), round(float(auc), 4),
                         round(lo, 4), round(hi, 4), round(float(ap), 4),
                         round(d, 4), round(dlo, 4), round(dhi, 4),
                         "excludes 0" if (dlo > 0 or dhi < 0) else "includes 0"])
    with io.open(RUN / "tables/subspace_metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "group", "n", "n_tb", "prevalence", "auroc", "auroc_lo",
                    "auroc_hi", "auprc", "delta_vs_baseline", "delta_lo", "delta_hi",
                    "paired_ci"])
        w.writerows(disc)

    print("\n(a) DISCRIMINATION on REPORT")
    print("  %-38s %-7s %7s %18s  %s" % ("arm", "group", "AUROC", "delta [95% CI]", ""))
    for r in disc:
        print("  %-38s %-7s %7.4f  %+7.4f [%+0.4f, %+0.4f]  %s"
              % (r[0], r[1], r[5], r[9], r[10], r[11], r[12]))

    # ---------------- B. thresholds / operating points ----------------
    thr_rows = []
    for arm, sc, thr_map in (
            ("A pooled threshold", "score_baseline",
             {g: P["pooled_threshold"] for g in scanners}),
            ("B scanner thresholds", "score_baseline",
             {g: P["scanner_thresholds"][str(g)] for g in scanners}),
            ("C offset-corrected, single threshold", "score_offset_corrected",
             {g: P["offset_corrected_pooled_threshold"] for g in scanners})):
        s = S[sc][rep]
        agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for g in scanners:
            m = scan[rep] == g
            o = op(y[rep][m], s[m], thr_map[g])
            for k in agg:
                agg[k] += o[k]
            sl, sh = wilson(o["tp"], o["tp"] + o["fn"])
            pl, ph = wilson(o["tn"], o["tn"] + o["fp"])
            thr_rows.append([arm, str(g), int(m.sum()), int(y[rep][m].sum()),
                             round(thr_map[g], 6), o["tp"], o["fp"], o["fn"], o["tn"],
                             round(o["sens"], 4), round(sl, 4), round(sh, 4),
                             round(o["spec"], 4), round(pl, 4), round(ph, 4),
                             round(o["ppv"], 4), round(o["npv"], 4),
                             round(o["referral"], 4)])
        n = sum(agg.values())
        sens = agg["tp"] / max(agg["tp"] + agg["fn"], 1)
        spec = agg["tn"] / max(agg["tn"] + agg["fp"], 1)
        sl, sh = wilson(agg["tp"], agg["tp"] + agg["fn"])
        pl, ph = wilson(agg["tn"], agg["tn"] + agg["fp"])
        thr_rows.append([arm, "POOLED", n, agg["tp"] + agg["fn"], float("nan"),
                         agg["tp"], agg["fp"], agg["fn"], agg["tn"],
                         round(sens, 4), round(sl, 4), round(sh, 4),
                         round(spec, 4), round(pl, 4), round(ph, 4),
                         round(agg["tp"] / max(agg["tp"] + agg["fp"], 1), 4),
                         round(agg["tn"] / max(agg["tn"] + agg["fn"], 1), 4),
                         round((agg["tp"] + agg["fp"]) / n, 4)])
    with io.open(RUN / "tables/threshold_report_metrics.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "group", "n", "n_tb", "threshold", "tp", "fp", "fn", "tn",
                    "sensitivity", "sens_lo", "sens_hi", "specificity", "spec_lo",
                    "spec_hi", "ppv", "npv", "referral_rate"])
        w.writerows(thr_rows)

    print("\n(b) OPERATING POINTS on REPORT")
    print("  %-38s %-7s %7s %7s %7s %7s %6s" %
          ("arm", "group", "sens", "spec", "refer", "ppv", "missed"))
    for r in thr_rows:
        print("  %-38s %-7s %7.3f %7.3f %7.3f %7.3f %6d"
              % (r[0], r[1], r[9], r[12], r[17], r[15], r[7]))

    # ---------------- C. acceptance criteria ----------------
    def worst(name):
        return min(roc_auc_score(y[rep][scan[rep] == g], S[name][rep][scan[rep] == g])
                   for g in scanners)
    w0, w1 = worst("score_baseline"), worst("score_P1_projected_frozen_head")
    d, dlo, dhi = C.paired_boot(y[rep], S["score_baseline"][rep],
                                S["score_P1_projected_frozen_head"][rep],
                                n_boot=N_BOOT, seed=SEED)
    c1 = bool(w1 > w0)
    c2 = bool(dlo > 0 or dhi < 0)
    verdict = {
        "criterion_1_worst_scanner_auroc_improves": c1,
        "worst_scanner_auroc_baseline": round(float(w0), 4),
        "worst_scanner_auroc_P1": round(float(w1), 4),
        "criterion_2_paired_pooled_ci_excludes_zero": c2,
        "paired_pooled_delta": round(d, 4),
        "paired_pooled_ci": [round(dlo, 4), round(dhi, 4)],
        "PREREGISTERED_OUTCOME": "ACCEPT" if (c1 and c2) else "REJECT",
    }
    # offset arm, reported separately; NOT covered by the preregistration
    do, dolo, dohi = C.paired_boot(y[rep], S["score_baseline"][rep],
                                   S["score_offset_corrected"][rep],
                                   n_boot=N_BOOT, seed=SEED)
    verdict["offset_correction_exploratory"] = {
        "paired_pooled_delta": round(do, 4), "ci": [round(dolo, 4), round(dohi, 4)],
        "ci_excludes_zero": bool(dolo > 0 or dohi < 0),
        "worst_scanner_auroc": round(float(worst("score_offset_corrected")), 4),
        "note": "not part of the preregistration; exploratory",
    }
    json.dump(verdict, io.open(RUN / "tables/acceptance_verdict.json", "w",
                               encoding="utf-8"), indent=2)
    print("\n(c) PREREGISTERED ACCEPTANCE CRITERIA")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
