# -*- coding: utf-8 -*-
"""Tests for the previous-TB subgroup analysis. CPU only; no GPU suite is launched.

    py -3 -m pytest artifacts/prevtb_subgroup_20261021_1055/tests -q
"""
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]

REP = json.loads((RUN / "tables/metric_reproduction.json").read_text(encoding="utf-8"))
MET = list(csv.DictReader(io.open(RUN / "tables/subgroup_metrics.csv", encoding="utf-8-sig")))
DIF = list(csv.DictReader(io.open(RUN / "tables/subgroup_difference_tests.csv",
                                  encoding="utf-8-sig")))
POW = json.loads((RUN / "tables/power_subgroup_study.json").read_text(encoding="utf-8"))
SCORES = ROOT / "artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv"
META = ROOT / "data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv"


def rows():
    return list(csv.DictReader(io.open(SCORES, encoding="utf-8-sig")))


def meta():
    return {r["pid"]: r for r in csv.DictReader(io.open(META, encoding="utf-8-sig"))}


# --------------------------------------------------------- provenance
def test_past_tb_is_documented_and_complete():
    readme = (ROOT / "data/raw/vietnam/README.md").read_text(encoding="utf-8")
    # the Dryad README uses a non-breaking space after the colon, so normalise whitespace
    flat = " ".join(readme.split())
    assert "past_tb (past history of TB): 0/No; 1/Yes" in flat
    vals = [r["past_tb"] for r in meta().values()]
    assert set(vals) == {"0", "1"}, "unexpected past_tb codes: %s" % set(vals)
    assert not any(v.strip() == "" for v in vals)


def test_past_tb_has_no_unknown_code_and_this_is_recorded():
    """dm has a 99 code; past_tb does not. The report must flag the consequence."""
    dm = {r["dm"] for r in meta().values()}
    assert "99" in dm
    f = REP["past_tb_field"]
    assert f["explicit_unknown_code"] is False
    assert "UNVERIFIED" in f["caveat"]


def test_one_image_per_patient_so_patient_level_bootstrap_is_valid():
    pid = [r["pid"] for r in rows()]
    assert len(pid) == 2708
    assert len(set(pid)) == len(pid)
    assert REP["patients"]["one_image_per_patient"] is True


# --------------------------------------------------------- counts and reproduction
def test_subgroup_counts_match_the_raw_data():
    m = meta()
    pid = [r["pid"] for r in rows()]
    y = np.array([int(m[p]["xpert_result"]) for p in pid])
    past = np.array([int(m[p]["past_tb"]) for p in pid])
    assert int(y.sum()) == 280
    assert int(((y == 0) & (past == 0)).sum()) == 1578
    assert int(((y == 0) & (past == 1)).sum()) == 850
    s = REP["subgroups"]
    assert s["subgroup_no_past_tb"]["n"] == 1858
    assert s["subgroup_past_tb"]["n"] == 1130
    assert s["positives_shared_by_both_subgroups"] == 280


def test_all_reproduction_checks_pass():
    assert REP["reproduction"]["all_pass"] is True
    for k, v in REP["reproduction"]["checks"].items():
        assert v["pass"], k


def test_reported_aurocs_recompute_from_raw_files():
    from sklearn.metrics import roc_auc_score
    m = meta()
    rs = rows()
    pid = [r["pid"] for r in rs]
    y = np.array([int(m[p]["xpert_result"]) for p in pid])
    past = np.array([int(m[p]["past_tb"]) for p in pid])
    ours = np.array([float(r["score_baseline"]) for r in rs])
    lunit = np.array([float(m[p]["lunit_3901"]) for p in pid])
    a = (y == 1) | ((y == 0) & (past == 0))
    b = (y == 1) | ((y == 0) & (past == 1))
    assert abs(roc_auc_score(y[a], ours[a]) - 0.7811) < 1e-4
    assert abs(roc_auc_score(y[b], ours[b]) - 0.6182) < 1e-4
    assert abs(roc_auc_score(y[a], lunit[a]) - 0.8170) < 1e-4
    assert abs(roc_auc_score(y[b], lunit[b]) - 0.7192) < 1e-4


# --------------------------------------------------------- the inference
def test_the_two_subgroups_share_every_positive():
    """If they did not, an unpaired comparison would have been acceptable."""
    assert REP["subgroups"]["positives_shared_by_both_subgroups"] == 280
    assert "shared" in REP["bootstrap"]["design"]


def test_difference_tests_report_intervals_not_bare_point_estimates():
    assert len(DIF) >= 3
    for r in DIF:
        assert r["estimate"] != ""
        assert r["ci95_lo"] != "" and r["ci95_hi"] != ""
        assert float(r["ci95_lo"]) < float(r["ci95_hi"])


def test_our_degradation_interval_excludes_zero():
    r = [x for x in DIF if x["contrast"] == "ours_noPastTB_minus_pastTB"][0]
    assert float(r["ci95_lo"]) > 0
    assert r["excludes_zero"] == "True"


def test_difference_in_differences_is_reported_with_its_sign_convention():
    r = [x for x in DIF if "difference_in_differences" in x["contrast"]][0]
    assert float(r["estimate"]) > 0
    assert "larger than the commercial comparator" in r["note"]


def test_subgroup_was_not_prespecified_and_this_is_recorded():
    p = REP["prespecification"]
    assert p["subgroup_prespecified"] is False
    assert len(p["variables_screened_in_that_diagnostic"]) >= 5


def test_refit_replicate_is_recorded_and_does_not_change_the_conclusion():
    r = REP["independent_refit_replicate"]
    assert r["scores_identical"] is False
    assert r["spearman_rank_correlation"] > 0.99
    # the refit's subgroup gap must agree with the primary one far inside the CI width
    gap_primary = REP["between_subgroup_differences"]["ours_noPastTB_minus_pastTB"]["estimate"]
    gap_refit = r["auroc_no_past_tb"] - r["auroc_past_tb"]
    assert abs(gap_refit - gap_primary) < 0.01


# --------------------------------------------------------- power
def test_power_is_recalculated_for_this_endpoint_not_reused():
    assert "between-subgroup AUROC difference" in POW["primary_endpoint"]
    assert POW["observed"]["se_gap"] > 0
    ns = {d["minimum_detectable_gap"]: d["required_participants"]
          for d in POW["primary_sample_size"]}
    assert 0.05 in ns
    # smaller detectable effects must need more participants
    ks = sorted(ns)
    assert all(ns[ks[i]] > ns[ks[i + 1]] for i in range(len(ks) - 1))


def test_power_states_assumptions_and_sensitivity_analyses():
    assert len(POW["assumptions"]) >= 5
    s = POW["sensitivity_analyses"]
    for k in ("previous_TB_fraction", "prevalence", "larger_SE_in_a_new_setting",
              "multiplicity_bonferroni_5_subgroup_variables"):
        assert k in s, k


# --------------------------------------------------------- safety
def test_internal_test_was_not_referenced_by_any_script_in_this_run():
    # a prose mention in a docstring is fine; what must not appear is a code reference
    for p in (RUN / "scripts").glob("*.py"):
        src = p.read_text(encoding="utf-8")
        for token in ('"internal_test"', "'internal_test'", "internal_test.csv"):
            assert token not in src, "%s references %s in code" % (p.name, token)


def test_no_script_in_this_run_trains_or_scores():
    banned = ("train_head", "stream_regions", "torch.nn.Linear", "backward()")
    for p in (RUN / "scripts").glob("*.py"):
        src = p.read_text(encoding="utf-8")
        for b in banned:
            assert b not in src, "%s contains %s" % (p.name, b)
