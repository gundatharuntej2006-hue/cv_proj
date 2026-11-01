# Protocol: second consecutive screening cohort

**Status: preregisterable draft. Not yet executable — the cohort does not exist in this
project and data access is the blocker.** §9 is the acquisition checklist.

This is a study-design document. It proposes no new architecture, no new training run and
no new model intervention, because the evidence says none of those is the limiting factor.

---

## 1. Why this study and not another model change

Four curated case-control cohorts now agree that regional max-pooling helps or is neutral
(Shenzhen +0.0279, Montgomery +0.0401, DB +0.0717, DA −0.0009). On the one consecutive
screening cohort the same configuration reaches **0.7242**, against **0.9497–0.9755** on
the curated sites and **0.7862** for a certified commercial product on identical patients.

Every intervention tried has moved the curated numbers far more than the screening number.
A fifth curated cohort would add precision to a question already answered and none to the
question that matters. **The limiting resource is screening-cohort data, not ideas.**

Two further results say the same thing from different directions: the preregistered
scanner-subspace removal was **rejected** (Δ −0.0002, CI [−0.0009, +0.0004]), and scanner
identity proved **not linearly removable** (0.9896 → 0.9118 decodability after stripping 44
of 768 directions, TB AUROC flat throughout). Representation-level intervention is not the
lever.

## 2. Primary endpoint

**Sensitivity at 70% specificity**, with specificity fixed on a locked development
partition and transferred unchanged.

That is the endpoint, not AUROC, because the intended use is **triage**: a fixed
confirmatory-testing budget, and the question is how many cases are caught within it. The
WHO target product profile for a TB triage test is 90% sensitivity at 70% specificity; the
current system reaches **0.59** on the Vietnam REPORT half and the commercial comparator
**0.65**, so both currently fail that bar and the endpoint is the one that shows it.

**Co-primary: AUROC**, reported with a 95% bootstrap CI, as the threshold-free summary.

Both are reported for the locked configuration and for a matched global-pooling baseline on
identical patients, paired.

## 3. Cohort requirements

| requirement | specification | why |
|---|---|---|
| **recruitment** | consecutive or otherwise documented representative enrolment at the point of care | the curated-to-screening gap is the object of study; a case-control set cannot measure it |
| **reference standard** | microbiological — Xpert Ultra or culture — on **all** participants, not only on those with abnormal radiographs | verification bias otherwise inflates sensitivity; this is the single most important requirement |
| **prevalence** | recorded, not controlled; expect 2–15% | a realistic negative-class spectrum is the point |
| **negative-class composition** | must include patients abnormal for reasons other than TB, recorded where known | our leading hypothesis for the gap |
| **case mix** | symptom status, prior TB, HIV and diabetes status, age band, sex where available | prespecified subgroups, §6 |
| **disease severity** | smear grade or Xpert semi-quantitative result where available | lets severity be separated from site |
| **acquisition metadata** | device/scanner identifier per image, plus site | enables the scanner analyses; **its absence does not disqualify a cohort** |
| **partitioning** | **patient-level**, with patient identifiers supplied | TBX11K ships none, which already blocks patient-grouped analysis there |
| **images** | frontal radiograph, DICOM preferred, original resolution | lets preprocessing be rerun rather than inherited |
| **one image per patient** | or, if repeats exist, flagged so resampling can be done by patient | otherwise intervals are too narrow |

**A second cohort from a different country and a different radiography vendor is worth
more than a larger cohort from the same setting**, because it tests whether the Vietnam
result is a property of that programme or of the method.

## 4. Sample size, with assumptions stated

Computed from the committed Vietnam REPORT per-case scores by resampling, **not asserted**.
Script: `scripts/power_planning.py`; output `tables/power_planning.json`.

**Method.** Empirical SE of the paired difference from 4,000 paired bootstrap resamples at
the observed n, extrapolated as SE ∝ 1/√n, then inverted:
n = n₀ · ((z_α + z_β)·SE₀ / Δ)².

**Assumptions, all of which must be checked against the real cohort before relying on
these numbers:** case mix, negative-class spectrum and score distribution similar to the
Vietnam cohort; one image per patient; paired design; two-sided α = 0.05, power 0.80;
prevalence 0.104; approximate normality of the difference; **no multiplicity adjustment**
— add one if more than one primary endpoint is tested.

Observed at n = 1,356 (141 TB): SE of the paired ΔAUROC **0.0119**, SE of the paired
Δsensitivity at 70% specificity **0.0390**.

**Paired ΔAUROC:**

| detectable Δ | required n | required TB cases |
|---|---|---|
| 0.0206 *(the observed offset-correction effect)* | **3,569** | 372 |
| 0.03 | 1,683 | 175 |
| 0.04 | 947 | 99 |
| 0.05 | 606 | 63 |
| 0.08 | 237 | 25 |

**Paired Δsensitivity at 70% specificity:**

| detectable Δ | required TB cases | implied n at prevalence 0.104 |
|---|---|---|
| 0.05 | 675 | 6,483 |
| 0.075 | 300 | 2,882 |
| 0.10 | **169** | **1,621** |
| 0.15 | 75 | 721 |

**What this tells the design.** Detecting the scanner-offset effect already observed would
need **about 3,600 participants** — 2.6× the Vietnam REPORT half, which is precisely why
that result came back inconclusive. A cohort of **roughly 1,600–3,000 participants with
all-participant microbiological confirmation** is the sensible target: powered for a
10-point sensitivity difference and for AUROC differences of 0.03 and above, which covers
every effect this project has measured except the smallest.

Below about 700 participants the study can only detect differences larger than anything
observed so far, and should not be run as a comparative trial.

**Caveat carried from the calculation:** these are powers to compare two arms on the same
patients. Estimating *absolute* performance to a given precision is a different and
generally smaller requirement.

## 5. Locking, and the order of operations

1. Lock the model weights, preprocessing, aggregation and **all thresholds** on existing
   permitted development data. Record hashes.
2. Write the analysis manifest — endpoints, subgroups, bootstrap seed and count, decision
   rules, exclusion rules — and hash it.
3. Only then score the new cohort, **once**.
4. No refitting, recalibration or threshold adjustment on the new cohort for the primary
   analysis. Recalibrated performance may be reported as a clearly labelled secondary.

This is the procedure already used twice in this project and it worked: in
`overnight_scanner_run_20261021_0208` the manifest was frozen before the single REPORT
pass, and a test asserts the manifest predates the results on disk.

## 6. Prespecified subgroups

Reported with intervals and **explicitly marked as underpowered unless §4 says otherwise**:
scanner/device; site; sex; age band; symptom status; prior TB; HIV and diabetes status
where recorded; smear grade or semi-quantitative Xpert result.

**Worst-subgroup performance is reported alongside the pooled figure in every table.** A
pooled gain with worst-subgroup degradation is a failure, not a success — the rule already
written into the scanner-subspace preregistration.

**No pooling across sites without also reporting per-site results.** Pooling cohorts with
different prevalence and score distributions can move AUROC through cross-cohort ranking
alone; this was measured directly, not assumed.

## 7. Calibration and referral burden, kept separate from discrimination

Reported in their own section, never merged into a single "performance" claim:

- **Discrimination:** AUROC, AUPRC with the baseline prevalence stated.
- **Calibration:** calibration slope and intercept, Brier score, a calibration plot.
  Calibrators fitted on development data only.
- **Referral burden:** referral rate, PPV, NPV, number needed to test per case found, and
  confirmatory tests consumed per 1,000 screened at the locked threshold.
- **Equity:** referral rate and specificity by subgroup, with the spread stated.

A threshold change cannot alter AUROC, and an additive per-group offset cannot alter
within-group AUROC. Any report that conflates these is wrong on its face.

## 8. What can be tested now, and what cannot

**Testable with current data — no new cohort needed:**
- DA/DB under any remaining representation (176 cases, already extracted at 12×12).
- Re-analysis of saved Vietnam per-case scores: subgroup error profiles, calibration,
  referral burden, power planning.
- Any purely internal ablation on TBX11K train/development.

**Requires a new cohort — not answerable with anything on disk:**
- Whether the curated-to-screening gap reproduces outside Vietnam.
- Whether the scanner-offset correction helps (needs ≈3,600 participants).
- Whether per-source thresholding is safe in deployment.
- Whether any configuration reaches the WHO triage bar in a real programme.
- Which of case mix, negative-class spectrum, acquisition, reference standard or selection
  drives the gap — this needs **two** screening cohorts differing in known ways, not one.

## 9. Data-acquisition checklist

Data access is the blocker, so this is the actionable list.

1. **Published consecutive screening cohorts with per-patient microbiological reference
   and image access.** The Vietnam Dryad cohort was found this way; others may exist and
   have not been systematically searched. Record what was searched and when.
2. **National TB programme active-case-finding datasets.** Typically require a data
   request and ethics approval; begin with the request template, not the model.
3. **Prospective collaboration with a screening programme.** Highest value, longest lead
   time. Needs a named clinical partner.
4. For each candidate, record **before** requesting access: enrolment design, reference
   standard and whether it was applied to all participants, prevalence, n and n positive,
   patient identifiers, acquisition metadata, licence and redistribution terms.
5. Check each candidate against §3 and against the §4 sample-size table, and **reject
   cohorts too small to answer the question** rather than running an underpowered study.
6. Record whether the cohort could plausibly have been seen by RAD-DINO's checkpoint
   selection. The model card does not name those datasets, so for public cohorts the
   honest answer will usually be **unknown**, and must be written as such.

**Do not substitute another architecture change for this work.** Attention-MIL was measured
and is worse; more training data is flat; subspace removal was rejected; the sweep is
complete. The next decisive result requires data, not a new model.

## 10. Interpretation rules, fixed in advance

- **A null result on an underpowered cohort is inconclusive, not evidence of no effect.**
  If the cohort cannot detect the prespecified effect per §4, the report must say
  "inconclusive, power inadequate" and must not say "no difference".
- A favourable point estimate with an interval including zero is **not** a positive result.
- Per-subgroup estimates below the measured noise floor are reported as noise regardless of
  sign. On the Vietnam cohort that floor was **±0.055 AUROC** between random halves of the
  same scanner's patients.
- No result from a single cohort establishes clinical readiness or generalisation.
- If the cohort was used for any configuration choice, it is exploratory and must be
  labelled so, permanently.
