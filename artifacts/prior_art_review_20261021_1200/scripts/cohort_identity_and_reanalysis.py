# -*- coding: utf-8 -*-
"""Confirm cohort identity against Codlin et al. 2025, and re-express our result under
their published stratification.

Read-only. No training, no scoring of new data, no threshold fitting, no internal_test.
Uses only committed per-case score files already in the repository.

Two constructions of a "previous-TB subgroup analysis" exist and they answer different
questions:

  A  FULL STRATIFICATION - split every participant by past_tb, so each stratum has its
     own positives and its own negatives. This is the conventional epidemiological
     subgroup analysis and it is what Codlin et al. 2025 published.

  B  SHARED POSITIVES - all 280 positives against the negatives of one stratum. This is
     what our earlier run computed. It asks how well active TB is separated from
     previous-TB negatives specifically.

Both are legitimate; they are not interchangeable. Under A the two strata are independent
samples, so the shared-positive bootstrap our earlier run needed is unnecessary there.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]

META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"
OURS = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"
N_BOOT = 10000
SEED = 20261021

PUBLISHED = {            # Codlin et al. 2025, PLOS Digit Health 4(4):e0000813, 2 dp
    ("overall", "v3.1.0.0"): 0.76, ("overall", "v3.9.0.1"): 0.78,
    ("past_tb", "v3.1.0.0"): 0.67, ("past_tb", "v3.9.0.1"): 0.73,
    ("no_past_tb", "v3.1.0.0"): 0.81, ("no_past_tb", "v3.9.0.1"): 0.81,
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    meta = {r["pid"]: r for r in csv.DictReader(io.open(META, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(io.open(OURS, encoding="utf-8-sig")))
    pid = [r["pid"] for r in rows]
    assert len(set(pid)) == len(pid) == 2708

    y = np.array([int(meta[p]["xpert_result"]) for p in pid])
    past = np.array([int(meta[p]["past_tb"]) for p in pid])
    ours = np.array([float(r["score_baseline"]) for r in rows])
    l39 = np.array([float(meta[p]["lunit_3901"]) for p in pid])
    l31 = np.array([float(meta[p]["lunit_3100"]) for p in pid])

    out = {"inputs": {str(META.relative_to(ROOT)).replace("\\", "/"): sha(META),
                      str(OURS.relative_to(ROOT)).replace("\\", "/"): sha(OURS)}}

    # ---------------- cohort identity ----------------
    ident = {
        "n_participants": int(len(pid)),
        "xpert_positive": int(y.sum()),
        "xpert_positivity_pct": round(float(100 * y.mean()), 2),
        "previous_tb_pct": round(float(100 * past.mean()), 2),
        "dryad_doi_in_local_readme": "10.5061/dryad.5tb2rbpcv",
        "local_readme_title": "Comparison of different Lunit INSIGHT CXR software "
                              "versions when reading chest radiographs for tuberculosis",
        "paper_reported": {"n": 2708, "positivity_pct": 10.3, "previous_tb_pct": 35.2,
                           "dryad_doi": "10.5061/dryad.5tb2rbpcv"},
        "scanner_counts": {str(v): int((np.array(
            [int(meta[p]["rad_system"]) for p in pid]) == v).sum()) for v in (1, 2, 3)},
    }
    ident["matches_on_n"] = ident["n_participants"] == 2708
    ident["matches_on_positivity"] = abs(ident["xpert_positivity_pct"] - 10.3) < 0.05
    ident["matches_on_previous_tb"] = abs(ident["previous_tb_pct"] - 35.2) < 0.05
    out["cohort_identity"] = ident

    # ---------------- reproduce the published Lunit AUCs ----------------
    strata = {"overall": np.ones(len(y), bool),
              "past_tb": past == 1, "no_past_tb": past == 0}
    repro = []
    for sname, mask in strata.items():
        for vname, sc in (("v3.1.0.0", l31), ("v3.9.0.1", l39)):
            a = roc_auc_score(y[mask], sc[mask])
            pub = PUBLISHED[(sname, vname)]
            repro.append({"stratum": sname, "version": vname,
                          "n": int(mask.sum()), "n_pos": int(y[mask].sum()),
                          "n_neg": int((y[mask] == 0).sum()),
                          "recomputed": round(float(a), 4),
                          "recomputed_2dp": round(float(a), 2),
                          "published_2dp": pub,
                          "match": bool(abs(round(float(a), 2) - pub) < 1e-9)})
    out["published_lunit_reproduction"] = {
        "construction": "FULL STRATIFICATION (interpretation A)",
        "rows": repro,
        "all_six_match_at_2dp": all(r["match"] for r in repro)}

    # ---------------- our model under BOTH constructions ----------------
    rng = np.random.default_rng(SEED)
    res = {"A_full_stratification": {}, "B_shared_positives": {}}

    # ---- A: independent strata, so an ordinary bootstrap per stratum, and an
    #         unpaired bootstrap for the difference between strata
    for label, sc in (("ours", ours), ("lunit_v3.9.0.1", l39), ("lunit_v3.1.0.0", l31)):
        blk = {}
        boots = {}
        for sname in ("past_tb", "no_past_tb"):
            m = strata[sname]
            idx_p = np.where(m & (y == 1))[0]
            idx_n = np.where(m & (y == 0))[0]
            pt = roc_auc_score(y[m], sc[m])
            bs = []
            for _ in range(N_BOOT):
                bp = rng.choice(idx_p, len(idx_p), replace=True)
                bn = rng.choice(idx_n, len(idx_n), replace=True)
                k = np.concatenate([bp, bn])
                bs.append(roc_auc_score(y[k], sc[k]))
            boots[sname] = np.array(bs)
            blk[sname] = {"n": int(m.sum()), "n_pos": int(len(idx_p)),
                          "n_neg": int(len(idx_n)),
                          "auroc": round(float(pt), 4),
                          "ci95": [round(x, 4) for x in ci(bs)]}
        d = boots["no_past_tb"] - boots["past_tb"]
        blk["degradation_noPast_minus_past"] = {
            "estimate": round(float(blk["no_past_tb"]["auroc"] - blk["past_tb"]["auroc"]), 4),
            "ci95": [round(x, 4) for x in ci(d)],
            "excludes_zero": bool(ci(d)[0] > 0 or ci(d)[1] < 0),
            "method": "unpaired bootstrap; under full stratification the two strata are "
                      "independent samples"}
        res["A_full_stratification"][label] = blk

    # difference in differences under A, resampling each of the four cells once
    idx = {(s, c): np.where(strata[s] & (y == c))[0]
           for s in ("past_tb", "no_past_tb") for c in (0, 1)}
    did = []
    for _ in range(N_BOOT):
        b = {k: rng.choice(v, len(v), replace=True) for k, v in idx.items()}
        kn = np.concatenate([b[("no_past_tb", 1)], b[("no_past_tb", 0)]])
        kp = np.concatenate([b[("past_tb", 1)], b[("past_tb", 0)]])
        o = roc_auc_score(y[kn], ours[kn]) - roc_auc_score(y[kp], ours[kp])
        l = roc_auc_score(y[kn], l39[kn]) - roc_auc_score(y[kp], l39[kp])
        did.append(o - l)
    oA = res["A_full_stratification"]["ours"]["degradation_noPast_minus_past"]["estimate"]
    lA = res["A_full_stratification"]["lunit_v3.9.0.1"][
        "degradation_noPast_minus_past"]["estimate"]
    res["A_full_stratification"]["difference_in_differences_ours_minus_lunit39"] = {
        "estimate": round(oA - lA, 4), "ci95": [round(x, 4) for x in ci(did)],
        "excludes_zero": bool(ci(did)[0] > 0 or ci(did)[1] < 0),
        "method": "bootstrap resampling all four stratum-by-class cells per replicate"}

    # ---- B: shared positives, as our earlier run computed (carried over for comparison)
    for label, sc in (("ours", ours), ("lunit_v3.9.0.1", l39)):
        blk = {}
        for sname, negmask in (("past_tb", (y == 0) & (past == 1)),
                               ("no_past_tb", (y == 0) & (past == 0))):
            m = (y == 1) | negmask
            blk[sname] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                          "n_neg": int((y[m] == 0).sum()),
                          "auroc": round(float(roc_auc_score(y[m], sc[m])), 4)}
        blk["degradation_noPast_minus_past"] = {
            "estimate": round(blk["no_past_tb"]["auroc"] - blk["past_tb"]["auroc"], 4),
            "note": "intervals for this construction are in "
                    "prevtb_subgroup_20261021_1055/tables/subgroup_difference_tests.csv; "
                    "they require the shared-positive bootstrap"}
        res["B_shared_positives"][label] = blk
    out["our_model_under_both_constructions"] = res

    json.dump(out, io.open(OUT / "tables/cohort_identity_and_reanalysis.json", "w",
                           encoding="utf-8"), indent=2)

    # ---------------- console ----------------
    print("COHORT IDENTITY")
    for k in ("n_participants", "xpert_positivity_pct", "previous_tb_pct",
              "matches_on_n", "matches_on_positivity", "matches_on_previous_tb"):
        print("  %-24s %s" % (k, ident[k]))
    print("  scanner counts          %s" % ident["scanner_counts"])
    print("\nPUBLISHED LUNIT AUCs, RECOMPUTED (full stratification)")
    print("  %-11s %-10s %6s %5s %10s %8s %6s" %
          ("stratum", "version", "n", "pos", "recomputed", "pub(2dp)", "match"))
    for r in repro:
        print("  %-11s %-10s %6d %5d %10.4f %8.2f %6s" %
              (r["stratum"], r["version"], r["n"], r["n_pos"], r["recomputed"],
               r["published_2dp"], r["match"]))
    print("  ALL SIX MATCH: %s" % out["published_lunit_reproduction"]["all_six_match_at_2dp"])

    print("\nOUR MODEL under A (full stratification, the published construction)")
    for lab in ("ours", "lunit_v3.9.0.1"):
        b = res["A_full_stratification"][lab]
        print("  %-14s past %.4f %s | no-past %.4f %s | degradation %+0.4f %s %s" %
              (lab, b["past_tb"]["auroc"], b["past_tb"]["ci95"],
               b["no_past_tb"]["auroc"], b["no_past_tb"]["ci95"],
               b["degradation_noPast_minus_past"]["estimate"],
               b["degradation_noPast_minus_past"]["ci95"],
               "EXCLUDES 0" if b["degradation_noPast_minus_past"]["excludes_zero"]
               else "includes 0"))
    d = res["A_full_stratification"]["difference_in_differences_ours_minus_lunit39"]
    print("  DiD (ours - Lunit v3.9) %+0.4f %s  %s" %
          (d["estimate"], d["ci95"],
           "EXCLUDES 0" if d["excludes_zero"] else "INCLUDES 0"))

    print("\nOUR MODEL under B (shared positives, our earlier construction)")
    for lab in ("ours", "lunit_v3.9.0.1"):
        b = res["B_shared_positives"][lab]
        print("  %-14s past %.4f | no-past %.4f | degradation %+0.4f" %
              (lab, b["past_tb"]["auroc"], b["no_past_tb"]["auroc"],
               b["degradation_noPast_minus_past"]["estimate"]))


if __name__ == "__main__":
    main()
