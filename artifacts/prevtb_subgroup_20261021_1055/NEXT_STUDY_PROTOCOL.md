# Pre-registered protocol: independent validation of the previous-TB subgroup effect

**Status: design. Not executable — no cohort has been secured.** Power is recalculated for
this protocol's own endpoint; no earlier sample-size figure is carried over.

---

## 1. Question

Is the previous-TB discrimination deficit observed in Vietnam — AUROC 0.7811 against
negatives without previous TB, 0.6182 against those with it, gap **0.163
[0.137, 0.189]** — a reproducible property of this system, or an artefact of one cohort
and one exploratory analysis?

Secondary: is our degradation larger than a commercial comparator's, as Vietnam suggested
(difference in differences +0.065 [0.040, 0.090])? Kagujje et al. found the effect is
**product-dependent** — CAD4TB degraded, qXR did not — so this is a live question, not a
foregone one.

## 2. Target population

Consecutive, or transparently and documentably sampled, participants in an active
case-finding or routine screening programme. Required: a **participant flow diagram**,
stated eligibility criteria, and an explicit sampling frame.

**A facility-based symptomatic-only sample is acceptable if documented as such** — that is
what Kagujje et al. had — but a community ACF cohort is preferred, and a cohort from a
different country and radiography vendor than Vietnam is worth more than a larger one from
the same setting.

The previous-TB subgroup must be identifiable prospectively, not reconstructed.

## 3. Endpoints

**Primary (one, chosen before any data are seen):**

> **The between-subgroup AUROC difference for our locked model:** AUROC(active TB vs
> negatives *without* previous TB) − AUROC(active TB vs negatives *with* previous TB).

Threshold-free, so it is unaffected by how thresholds transfer, and it is the statistic
that distinguishes a ranking failure from a calibration failure. Estimated with the same
stratified bootstrap used here — positives shared within replicate, negative strata
resampled independently, 10,000 replicates, patient-level.

**Co-primary, with the multiplicity consequence accepted (see §4):**

> **Sensitivity at 70% specificity**, overall and within each previous-TB stratum, at a
> threshold fixed on development/calibration data and transferred unchanged.

This is the clinically meaningful operating-point endpoint — the WHO target product
profile for a TB triage test is 90% sensitivity at 70% specificity. Two co-primaries
require an adjusted alpha; §4 uses 0.025 each.

**Secondary:** the difference in differences against a commercial comparator on identical
participants; AUPRC with prevalence stated; calibration; referrals per 1,000; performance
in the unknown-history stratum if one exists.

## 4. Minimum meaningful effect, fixed now

**Primary: a between-subgroup AUROC gap of 0.05.**

Justification, given before any new data: Vietnam's observed gap is 0.163 and the
commercial comparator's is 0.098, so 0.05 is well below both and would still be detected.
A gap smaller than 0.05 would be hard to distinguish from the ±0.055 per-subgroup noise
floor measured on the Vietnam cohort between random halves, and would not on its own
justify a subgroup-specific deployment policy.

**Co-primary: a 10-percentage-point difference in sensitivity at 70% specificity** between
strata. A smaller difference is within the width of most sensitivity intervals achievable
at realistic cohort sizes.

**These are fixed and will not be revised after seeing the cohort.**

## 5. Sample size and power

Computed in `scripts/power_for_subgroup_study.py`; full output
`tables/power_subgroup_study.json`. Method: empirical SE from the 10,000-replicate
stratified bootstrap on Vietnam (**SE of the gap = 0.0131**, SE of the difference in
differences = 0.0127), extrapolated as SE ∝ 1/√n at fixed design, then inverted:
n = n₀·((z_α + z_β)·SE₀/δ)².

**Assumptions, each of which must be re-checked against the real cohort:** ~10%
prevalence; **~35% of negatives with previous TB**; one image per patient; similar
class-conditional score distributions; two-sided α = 0.05, power 0.80; approximate
normality; and the Vietnam SE is itself an estimate carrying its own uncertainty.

**Primary endpoint:**

| minimum detectable gap | participants | TB cases | previous-TB negatives |
|---|---|---|---|
| **0.05** | **1,467** | 152 | 461 |
| 0.075 | 652 | 68 | 205 |
| 0.10 | 367 | 38 | 116 |

**Secondary (difference in differences):** 1,366 participants for 0.05; 808 for the 0.065
observed in Vietnam.

### Sensitivity analyses

| varied assumption | participants for a 0.05 gap |
|---|---|
| previous TB 35% of negatives (Vietnam) | 1,467 |
| **previous TB 24% (Kagujje et al.)** | **2,139** |
| previous TB 15% | 3,422 |
| previous TB 10% | 5,133 |
| prevalence 10.3% (Vietnam) | 1,472 |
| prevalence 5% | 3,032 |
| prevalence 3% | 5,054 |
| SE 1.25× larger in a new setting | 2,291 |
| SE 1.5× larger | 3,299 |
| SE 2× larger | 5,865 |
| **Bonferroni across the 5 variables originally screened** (α = 0.01) | **2,182** |

**Recommended target: 2,500–3,500 participants**, which covers the Bonferroni-adjusted
requirement (2,182), the Kagujje-like previous-TB fraction (2,139) and a moderately larger
SE (2,291), with **15–25% attrition** budgeted on top — so **enrol ~3,500–4,500**.

**This supersedes the "2,500–4,000" figure from the earlier run**, which was computed for
a different endpoint (precision of a single subgroup AUROC). The ranges happen to overlap;
the derivation is different and the one above is the one that applies here.

**Operating-point sensitivity is sized separately and is the binding constraint if
prevalence is low:** at 10% prevalence, detecting a 10-point sensitivity difference
between strata needs on the order of 150–200 TB cases *within each stratum*, i.e. roughly
3,000–4,000 participants at a 35% previous-TB fraction, and considerably more at 24%. If
prevalence is below 5%, the co-primary should be demoted to secondary and the study
powered on the primary alone.

## 6. Partitions

| role | data | rule |
|---|---|---|
| model development | TBX11K train | unchanged, frozen before the study |
| calibration and threshold selection | TBX11K development + calibration | unchanged |
| **final evaluation** | **the new cohort, scored once** | patient-level, no overlap |

**Patient-level partitioning throughout**, with pseudonymous identifiers supplied so
de-duplication can be verified rather than assumed. A site-held-out split is preferred if
the cohort spans institutions.

**Previously exposed and not relabellable as validation:** TBX11K train/development/
calibration; Vietnam TUNE **and** REPORT; Shenzhen; Montgomery; DA and DB. RAD-DINO
checkpoint exposure to any public benchmark is **unknown**.

**TBX11K `internal_test` (901) remains unread** and is not part of this study.

## 7. Reference standard and verification bias

**Preferred: microbiological confirmation for every enrolled participant**, independent of
the AI output and of symptoms.

If universal confirmation is infeasible, in descending order of acceptability:

1. Random sample of the screen-negative stratum, tested regardless, with the sampling
   fraction prespecified and inverse-probability weighting applied.
2. Documented 6–12 month follow-up of untested participants for incident TB.
3. Report sensitivity as an **upper bound**, stated in the abstract, not only in the
   limitations.

**Why this matters more than usual here.** If testing is triggered by an abnormal
radiograph, then in the previous-TB stratum — where radiographs are abnormal for reasons
other than active disease — the triggering is differential by subgroup. That biases the
subgroup comparison itself, not merely the absolute estimates.

**Xpert Ultra "trace" results must be adjudicated and the rule prespecified.** Persistent
or trace positivity in previously treated patients is a known phenomenon; whether trace is
counted positive materially affects this exact subgroup. Whether Vietnam adjudicated it is
unknown.

## 8. Metadata requirements

Required: previous TB history — **record-verified where possible, with the ascertainment
method recorded**, and an explicit **unknown** category; microbiological result; image
view; de-identification and provenance.

Strongly preferred: scanner/vendor/site; image quality flags; symptoms and risk factors;
smear grade or semi-quantitative Xpert; age band and sex where ethically and legally
appropriate; prior treatment completion date.

**The unknown category is not optional.** Vietnam's `past_tb` has no unknown code, so it
cannot be determined whether unknown histories were coded "no" — a limitation this study
must not inherit.

## 9. Handling missing and unknown previous-TB history

Primary analysis: complete cases with a documented history, either way.

Prespecified sensitivity analyses: (a) unknowns assigned to "no previous TB"; (b) unknowns
assigned to "previous TB"; (c) unknowns as a third stratum reported separately. If (a) and
(b) give materially different conclusions, the primary result is reported as
**inconclusive with respect to history ascertainment**.

## 10. Locking and external-validation rules

1. Lock weights, preprocessing, aggregation, label mapping and **all thresholds** on
   permitted development data. Record hashes.
2. Write and hash the analysis manifest — endpoints, subgroups, bootstrap seed and
   replicate count, decision rules, exclusions, missing-data handling.
3. Score the new cohort **once**.
4. **No model, preprocessing or threshold change after test results are seen.** Any such
   change converts the study to exploratory, permanently and in writing.

A test must assert that the manifest predates the results on disk. This project has done
that twice; keep it.

## 11. Reporting

AUROC and AUPRC with prevalence; sensitivity and specificity at the locked threshold;
calibration slope, intercept and Brier; referrals per 1,000 and confirmatory tests
consumed; every subgroup estimate with an interval; **missingness and model failures
(unreadable images, segmentation failures) reported explicitly rather than dropped**;
worst-subgroup performance beside every pooled figure.

Discrimination, calibration and referral burden are reported in separate sections. A
threshold change cannot alter AUROC; an additive per-group offset cannot alter within-group
AUROC.

## 12. Governance

Ethics/IRB approval covering secondary AI analysis; a data-sharing agreement with
publication and authorship terms; de-identification standard including DICOM headers and
burnt-in text; storage, encryption and retention; local clinical oversight; and an explicit
commitment that no patient-level data leaves the agreed environment. **Offer to run the
analysis inside the partner's environment** — the pipeline needs under 1 GB of VRAM and
minutes of compute.

## 13. Decision categories, fixed in advance

| outcome | definition |
|---|---|
| **Success** | primary gap ≥ 0.05 with a 95% CI excluding zero, and no protocol deviation |
| **Negative** | primary gap < 0.05 with a CI that excludes 0.05, i.e. a meaningful effect is ruled out |
| **Inconclusive** | the CI contains both 0 and 0.05, **or** the achieved sample is below the §5 requirement — an underpowered null is inconclusive, never evidence of no effect |
| **Protocol deviation** | any change to model, preprocessing or threshold after results are seen; any unplanned endpoint change; any failure of the locking procedure. Reported as such, and the study is reclassified exploratory |

## 14. What this study cannot settle

Causation. Even a clean replication would show that previous TB *predicts* worse ranking,
not that post-treatment radiographic change *causes* it; history correlates with age,
prior care-seeking, comorbidity and repeated imaging. Establishing the mechanism needs
radiologist-adjudicated post-TB findings, which belong in a separate study.
