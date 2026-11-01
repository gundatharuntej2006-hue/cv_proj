# -*- coding: utf-8 -*-
"""Phases 2 and 3: check every numeric claim in the prior run, and reproduce the power
calculations from their inputs rather than from their printed outputs.

Read-only. Nothing is trained, scored or fitted. No existing file is modified.
TBX11K internal_test is not referenced.
"""
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
PREV = ROOT / "artifacts/prevtb_subgroup_20261021_1055"

Z_A, Z_B = 1.959964, 0.841621


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rep = json.loads((PREV / "tables/metric_reproduction.json").read_text(encoding="utf-8"))
    pow_sub = json.loads((PREV / "tables/power_subgroup_study.json").read_text(encoding="utf-8"))
    met = list(csv.DictReader(io.open(PREV / "tables/subgroup_metrics.csv",
                                      encoding="utf-8-sig")))
    dif = list(csv.DictReader(io.open(PREV / "tables/subgroup_difference_tests.csv",
                                      encoding="utf-8-sig")))
    report = " ".join((PREV / "PREVIOUS_TB_SUBGROUP_REPORT.md")
                      .read_text(encoding="utf-8").split())
    protocol = " ".join((PREV / "NEXT_STUDY_PROTOCOL.md")
                        .read_text(encoding="utf-8").split())

    checks = []

    def chk(name, expected, actual, ok=None, note=""):
        ok = (expected == actual) if ok is None else ok
        checks.append({"check": name, "expected": expected, "actual": actual,
                       "pass": bool(ok), "note": note})

    # ---------------- Phase 2: statistical claims ----------------
    chk("unique patients = 2708", 2708, rep["patients"]["unique_pid"])
    chk("one image per patient", True, rep["patients"]["one_image_per_patient"])
    chk("past_tb fully populated (0 blanks)", 0, rep["past_tb_field"]["missing_or_blank"])
    chk("past_tb has NO explicit unknown code", False,
        rep["past_tb_field"]["explicit_unknown_code"])
    chk("past_tb self-reported, not record-verified", True,
        "Self-reported" in rep["past_tb_field"]["caveat"],
        note=rep["past_tb_field"]["caveat"][:90])

    pe = rep["point_estimates"]
    chk("ours no-previous-TB = 0.7811", 0.7811, round(pe["ours_no_past_tb"], 4))
    chk("ours previous-TB = 0.6182", 0.6182, round(pe["ours_past_tb"], 4))
    chk("Lunit no-previous-TB = 0.8170", 0.817, round(pe["lunit_no_past_tb"], 4))
    chk("Lunit previous-TB = 0.7192", 0.7192, round(pe["lunit_past_tb"], 4))

    rf = rep["independent_refit_replicate"]
    chk("Phase 2 refit no-previous-TB = 0.7822", 0.7822, rf["auroc_no_past_tb"])
    chk("Phase 2 refit previous-TB = 0.6173", 0.6173, rf["auroc_past_tb"])
    chk("refit is a SEPARATE fit, not mixed into the primary comparison", True,
        rf["scores_identical"] is False,
        note="phases must not be compared against each other in one contrast")

    bd = rep["between_subgroup_differences"]
    o = bd["ours_noPastTB_minus_pastTB"]
    l = bd["lunit_noPastTB_minus_pastTB"]
    d = bd["difference_in_differences_ours_minus_lunit"]
    chk("our degradation = 0.1629", 0.1629, round(o["estimate"], 4))
    chk("our degradation CI = [0.1372, 0.1890]", [0.1372, 0.189],
        [round(x, 4) for x in o["ci95"]])
    chk("Lunit degradation = 0.0978", 0.0978, round(l["estimate"], 4))
    chk("Lunit degradation CI = [0.0770, 0.1196]", [0.077, 0.1196],
        [round(x, 4) for x in l["ci95"]])
    chk("difference-in-differences = +0.0651", 0.0651, round(d["estimate"], 4))
    chk("DiD CI = [0.0399, 0.0900]", [0.0399, 0.09], [round(x, 4) for x in d["ci95"]])
    chk("DiD arithmetic: ours - Lunit", round(o["estimate"] - l["estimate"], 4),
        round(d["estimate"], 4))

    ns = rep["negative_stratum_separability"]
    chk("negative-only separation = 0.6914", 0.6914, round(ns["estimate"], 4))
    chk("negative-only separation CI = [0.6698, 0.7126]", [0.6698, 0.7126],
        [round(x, 4) for x in ns["ci95"]])

    chk("subgroup NOT prespecified", False, rep["prespecification"]["subgroup_prespecified"])
    chk("five clinical variables screened", 5,
        len(rep["prespecification"]["variables_screened_in_that_diagnostic"]))
    chk("bootstrap shares the positive resample within each replicate", True,
        "shared by both subgroup AUROCs" in rep["bootstrap"]["design"],
        note=rep["bootstrap"]["design"])
    chk("positives shared by both subgroups = 280", 280,
        rep["subgroups"]["positives_shared_by_both_subgroups"])
    chk("report states performance without previous TB is ALSO below the comparator", True,
        "0.7811 even among negatives" in report and "0.8170" in report,
        note="guards against implying the subgroup explains the whole gap")
    chk("report declines a causal claim", True,
        "Not causal" in report or "not demonstrated" in report)

    # ---------------- Phase 3: reproduce the power numbers from inputs ----------------
    se_gap = pow_sub["observed"]["se_gap"]
    se_did = pow_sub["observed"]["se_did"]
    n0 = pow_sub["design_constants_from_vietnam"]["n0"]
    prev = pow_sub["design_constants_from_vietnam"]["prevalence"]
    pastfrac = pow_sub["design_constants_from_vietnam"]["previous_TB_fraction_among_negatives"]

    def need_f(delta, se, za=Z_A):
        """raw float, matching the original script before its single ceil"""
        return n0 * ((za + Z_B) * se / abs(delta)) ** 2

    def need(delta, se, za=Z_A):
        return int(np.ceil(need_f(delta, se, za)))

    for row in pow_sub["primary_sample_size"]:
        d_ = row["minimum_detectable_gap"]
        chk("primary n reproduces for gap %.4g" % d_, row["required_participants"],
            need(d_, se_gap), note="n = n0*((za+zb)*SE/delta)^2 with SE=%.5f" % se_gap)
    for row in pow_sub["secondary_sample_size"]:
        d_ = row["minimum_detectable_did"]
        chk("secondary (DiD) n reproduces for %.4g" % d_, row["required_participants"],
            need(d_, se_did))

    sens = pow_sub["sensitivity_analyses"]
    chk("Bonferroni(5) n reproduces", sens["multiplicity_bonferroni_5_subgroup_variables"]
        ["required_participants_for_gap_0.05"], need(0.05, se_gap, za=2.575829))
    for r in sens["larger_SE_in_a_new_setting"]:
        chk("SE x%.2f n reproduces" % r["SE_multiplier"],
            r["required_participants_for_gap_0.05"], need(0.05, se_gap * r["SE_multiplier"]))
    for r in sens["prevalence"]:
        chk("prevalence %.3f n reproduces" % r["prevalence"],
            r["required_participants_for_gap_0.05"],
            int(np.ceil(need_f(0.05, se_gap) * (prev / r["prevalence"]))))
    for r in sens["previous_TB_fraction"]:
        f_ = r["fraction_of_negatives_with_previous_TB"]
        chk("previous-TB fraction %.2f n reproduces" % f_,
            r["required_participants_for_gap_0.05"],
            int(np.ceil(need_f(0.05, se_gap) * (pastfrac / f_))))

    # the protocol's recommended range, and what it is powered for
    rec = re.search(r"Recommended target[^\d]{0,20}([\d,]+)\s*[–-]\s*([\d,]+)",
                    protocol)
    chk("protocol states a recommended range", True, rec is not None,
        note=rec.group(0) if rec else "not found")
    needs = {r["minimum_detectable_gap"]: r["required_participants"]
             for r in pow_sub["primary_sample_size"]}
    covered = {
        "gap_0.05_unadjusted": needs[0.05],
        "bonferroni_5": sens["multiplicity_bonferroni_5_subgroup_variables"]
                            ["required_participants_for_gap_0.05"],
        "previous_TB_fraction_0.24": [r["required_participants_for_gap_0.05"]
                                      for r in sens["previous_TB_fraction"]
                                      if r["fraction_of_negatives_with_previous_TB"] == 0.24][0],
        "SE_1.25x": [r["required_participants_for_gap_0.05"]
                     for r in sens["larger_SE_in_a_new_setting"]
                     if r["SE_multiplier"] == 1.25][0],
    }
    chk("recommended 2,500-3,500 covers every binding scenario in the table", True,
        all(v <= 2500 for v in covered.values()),
        note=json.dumps(covered))

    # CIDRZ cohort adequacy
    chk("CIDRZ 1,884 is BELOW the requirement at a 24% previous-TB fraction", True,
        1884 < covered["previous_TB_fraction_0.24"],
        note="1884 vs %d required" % covered["previous_TB_fraction_0.24"])

    # ---------------- the other power script, and endpoint compatibility -------------
    other = ROOT / "artifacts/doc_finalisation_20261021_0930/tables/power_planning.json"
    other_j = json.loads(other.read_text(encoding="utf-8")) if other.exists() else None
    endpoint_note = {
        "scripts_found": {
            "prevtb_subgroup_20261021_1055/scripts/power_for_subgroup_study.py":
                "sizes the BETWEEN-SUBGROUP AUROC difference - this is what "
                "NEXT_STUDY_PROTOCOL.md is built on",
            "doc_finalisation_20261021_0930/scripts/power_planning.py":
                "sizes a DIFFERENT endpoint: the paired AUROC difference between two model "
                "arms on the whole cohort, and the precision of a single-arm AUROC",
        },
        "the_brief_named_power_planning_py": True,
        "resolution": "the brief refers to scripts/power_planning.py; the script governing "
                      "the protocol under review is power_for_subgroup_study.py. Both were "
                      "inspected. They size different endpoints and their outputs must not "
                      "be combined.",
        "doc_finalisation_endpoint": (other_j or {}).get("endpoint"),
        "doc_finalisation_requirement_for_its_own_endpoint": (
            (other_j or {}).get("auroc_sample_size", [{}])[0]),
    }

    out = {
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
                   for p in sorted(PREV.rglob("*"))
                   if p.is_file() and p.suffix in (".md", ".json", ".csv", ".py")},
        "phase2_and_3_checks": checks,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["pass"]),
        "all_pass": all(c["pass"] for c in checks),
        "power_script_endpoint_audit": endpoint_note,
        "multiplicity_assessment": {
            "selection": "one subgroup variable chosen after screening five "
                         "(past_tb, dm, symptom count, contact, age_group)",
            "comparisons_actually_reported": [
                "ours: AUROC in each of 2 subgroups",
                "Lunit: AUROC in each of 2 subgroups",
                "between-subgroup difference for ours",
                "between-subgroup difference for Lunit",
                "difference in differences",
                "negative-only separability",
            ],
            "bonferroni_across_5_variables_is": "a first-order correction for the SELECTION "
                                                "of the variable, not for the six contrasts "
                                                "reported from it",
            "bootstrap_p_resolution_floor": 1.0 / 10000,
            "consequence": "the reported 'p < 0.0001' is at the resolution floor of a "
                           "10,000-replicate percentile bootstrap. Bonferroni across five "
                           "still leaves it below 0.0005, so the conclusion is unchanged, "
                           "but a properly selective-inference treatment was NOT performed "
                           "and the intervals are not selection-corrected.",
            "verdict": "QUALIFY, do not drop. The claim 'survives Bonferroni across five' "
                       "is defensible as a rough check but must not be presented as a "
                       "complete multiplicity treatment.",
        },
    }
    json.dump(out, io.open(OUT / "tables/claim_verification.json", "w", encoding="utf-8"),
              indent=2)

    print("checks: %d, failed: %d" % (out["n_checks"], out["n_failed"]))
    for c in checks:
        if not c["pass"]:
            print("  FAIL %-58s expected=%s actual=%s" % (c["check"], c["expected"],
                                                          c["actual"]))
    print("\npower script endpoints:")
    for k, v in endpoint_note["scripts_found"].items():
        print("  %s\n     -> %s" % (k, v))


if __name__ == "__main__":
    main()
