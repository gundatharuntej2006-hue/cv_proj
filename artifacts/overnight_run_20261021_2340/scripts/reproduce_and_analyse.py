# -*- coding: utf-8 -*-
"""Priority 3 and Priority 5: reproduce reported baselines and analyse errors by scanner.

Nothing is fitted here. Every number comes from a committed case-level score file, and
every reported figure it is compared against comes from a committed result JSON/CSV.

Threshold policy: the only operating point used is the one taken at 70% specificity on the
Vietnam TUNE half. TUNE is the documented development partition. No threshold is chosen
using REPORT labels, and no threshold is chosen using Shenzhen or Montgomery labels.
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
N_BOOT = 4000


def boot_ci(y, p, seed=0, n=N_BOOT):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    vals = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if len(np.unique(y[k])) < 2:
            continue
        vals.append(roc_auc_score(y[k], p[k]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def row(cohort, arm, subset, y, p, reported=None, seed=0):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    lo, hi = boot_ci(y, p, seed)
    delta = "" if reported is None else round(auc - reported, 6)
    return [cohort, arm, subset, len(y), int(y.sum()), int((y == 0).sum()),
            round(float(y.mean()), 4), round(float(auc), 6), round(float(ap), 6),
            round(lo, 4), round(hi, 4),
            "" if reported is None else reported, delta,
            "MATCH" if reported is not None and abs(round(auc, 4) - reported) < 5e-5
            else ("" if reported is None else "DIFFERS")]


def main():
    out = []

    # ---------- Vietnam, focal pooling (12x12 max, seeds 17/29/43/61/79) ----------
    ga = list(csv.DictReader(io.open(
        ROOT / "artifacts/p2/audit/vietnam_case_level_gate_audit.csv", encoding="utf-8-sig")))
    y = np.array([int(r["xpert_result"]) for r in ga])
    s = np.array([float(r["classifier_score"]) for r in ga])
    half = np.array([r["split_half"] for r in ga])
    scan = np.array([int(r["rad_system"]) for r in ga])
    ood = np.array([float(r["ood_score"]) for r in ga])

    mil = json.load(io.open(ROOT / "artifacts/p2/regional/mil_final.json", encoding="utf-8"))
    rep_tune = mil["results"]["Vietnam TUNE"]["auc"]
    rep_report = mil["results"]["Vietnam REPORT"]["auc"]

    out.append(row("Vietnam", "C focal 12x12 max", "ALL (2708)", y, s))
    out.append(row("Vietnam", "C focal 12x12 max", "TUNE", y[half == "TUNE"], s[half == "TUNE"],
                   rep_tune))
    out.append(row("Vietnam", "C focal 12x12 max", "REPORT", y[half == "REPORT"],
                   s[half == "REPORT"], rep_report))
    for v in (1, 2, 3):
        m = scan == v
        out.append(row("Vietnam", "C focal 12x12 max", "rad_system=%d" % v, y[m], s[m]))
        for h in ("TUNE", "REPORT"):
            mm = m & (half == h)
            out.append(row("Vietnam", "C focal 12x12 max",
                           "rad_system=%d / %s" % (v, h), y[mm], s[mm]))

    # the commercial comparator on the identical patients, from the Dryad columns
    meta = {str(r["pid"]): r for r in csv.DictReader(io.open(
        ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv", encoding="utf-8-sig"))}
    lun = np.array([float(meta[r["pid"]]["lunit_3901"]) for r in ga])
    ve = json.load(io.open(ROOT / "artifacts/p2/external/vietnam_eval.json", encoding="utf-8"))
    rep_lun = [r for r in ve["results"] if r["variant"] == "lunit_3901"][0]["auc"]
    out.append(row("Vietnam", "Lunit INSIGHT 3.9.0.1", "ALL (2708)", y, lun, rep_lun))
    for v in (1, 2, 3):
        m = scan == v
        out.append(row("Vietnam", "Lunit INSIGHT 3.9.0.1", "rad_system=%d" % v, y[m], lun[m]))

    # ---------- Shenzhen / Montgomery, three arms ----------
    nlm = list(csv.DictReader(io.open(
        ROOT / "artifacts/p2/audit/cross_dataset_case_predictions.csv", encoding="utf-8-sig")))
    cols = [c for c in nlm[0] if c not in ("image_id", "cohort", "label")]
    rep = {}
    for r in csv.DictReader(io.open(
            ROOT / "artifacts/p2/audit/cross_dataset_metrics.csv", encoding="utf-8-sig")):
        rep[(r.get("cohort"), r.get("arm"))] = r
    for coh in ("Shenzhen", "Montgomery"):
        sub = [r for r in nlm if r["cohort"] == coh]
        yy = np.array([int(r["label"]) for r in sub])
        for c in cols:
            pp = np.array([float(r[c]) for r in sub])
            arm = c[len("score_"):]
            k = rep.get((coh, arm))
            reported = float(k["auroc"]) if k else None
            out.append(row(coh, c, "all", yy, pp, reported))

    hdr = ["cohort", "arm", "subset", "n", "n_pos", "n_neg", "prevalence", "auroc_recomputed",
           "auprc_recomputed", "ci95_lo", "ci95_hi", "auroc_as_reported", "difference",
           "verdict"]
    with io.open(RUN / "baseline_reproduction.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(out)
    print("wrote baseline_reproduction.csv (%d rows)" % len(out))
    for r in out:
        print("  %-10s %-22s %-24s n=%-5d auc=%.4f %s" % (r[0], r[1], r[2], r[3], r[7], r[13]))

    # ---------- Priority 5.1 and 5.2: errors and score distributions by scanner ----------
    is_tune = half == "TUNE"
    thr = float(np.quantile(s[is_tune & (y == 0)], 0.70))

    err = []
    for v in (0, 1, 2, 3):
        m = np.ones(len(y), bool) if v == 0 else (scan == v)
        name = "ALL" if v == 0 else "rad_system=%d" % v
        for h in ("ALL", "TUNE", "REPORT"):
            mm = m if h == "ALL" else (m & (half == h))
            if mm.sum() == 0:
                continue
            yy, ss = y[mm], s[mm]
            flag = ss >= thr
            tp = int((flag & (yy == 1)).sum())
            fp = int((flag & (yy == 0)).sum())
            fn = int((~flag & (yy == 1)).sum())
            tn = int((~flag & (yy == 0)).sum())
            err.append([
                name, h, int(mm.sum()), int(yy.sum()), round(float(yy.mean()), 4),
                round(thr, 6), tp, fp, fn, tn,
                round(tp / max(tp + fn, 1), 4), round(tn / max(tn + fp, 1), 4),
                round(tp / max(tp + fp, 1), 4),
                round(float(np.median(ss[yy == 1])), 4) if (yy == 1).any() else "",
                round(float(np.median(ss[yy == 0])), 4) if (yy == 0).any() else "",
                round(float(np.mean(ood[mm] > 0.7)), 4),
            ])
    with io.open(RUN / "error_analysis.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "half", "n", "n_tb", "prevalence", "threshold",
                    "true_pos", "false_pos", "false_neg", "true_neg",
                    "sensitivity", "specificity", "precision",
                    "median_score_tb", "median_score_non_tb", "ood_reject_rate"])
        w.writerows(err)
    print("wrote error_analysis.csv (%d rows)" % len(err))

    # score distribution detail, saved as JSON for the report
    dist = {"threshold_source": "70% specificity on the Vietnam TUNE half, non-TB cases",
            "threshold": round(thr, 6), "by_scanner": {}}
    for v in (1, 2, 3):
        m = scan == v
        q = lambda a: [round(float(x), 4) for x in np.percentile(a, [5, 25, 50, 75, 95])]
        dist["by_scanner"][str(v)] = {
            "n": int(m.sum()), "n_tb": int(y[m].sum()),
            "prevalence": round(float(y[m].mean()), 4),
            "auroc": round(float(roc_auc_score(y[m], s[m])), 4),
            "score_percentiles_all": q(s[m]),
            "score_percentiles_tb": q(s[m & (y == 1)]),
            "score_percentiles_non_tb": q(s[m & (y == 0)]),
            "referral_rate_at_shared_threshold": round(float((s[m] >= thr).mean()), 4),
            "local_threshold_for_70pct_spec": round(
                float(np.quantile(s[m & (y == 0)], 0.70)), 6),
            "sensitivity_at_shared_threshold": round(
                float((s[m & (y == 1)] >= thr).mean()), 4),
            "sensitivity_at_local_threshold": round(
                float((s[m & (y == 1)] >= np.quantile(s[m & (y == 0)], 0.70)).mean()), 4),
            "ood_reject_rate": round(float((ood[m] > 0.7).mean()), 4),
        }
    (RUN / "score_distribution_metrics.json").write_text(
        json.dumps(dist, indent=2), encoding="utf-8")
    print(json.dumps(dist, indent=2))


if __name__ == "__main__":
    main()
