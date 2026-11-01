# Task 2 — Scanner-specific referral thresholds

**Design frozen in `FROZEN_MANIFEST.md` §2a before REPORT was scored. Thresholds were
fitted on TUNE only, by empirical quantile, never by inspecting REPORT.**

Tables: `tables/baseline_by_scanner.csv`, `tables/threshold_calibration.csv`,
`tables/threshold_bootstrap.csv`, `tables/threshold_report_metrics.csv`

---

## The one thing to read first

**A threshold cannot change AUROC.** Ranking is unaffected by where the cut is placed, so
nothing in this document is a discrimination improvement, and none is claimed. What changes
is who gets referred.

## 1. The problem, measured on TUNE

Three scanners are present in the cohort manifest (`rad_system` ∈ {1, 2, 3}); the
previously reported two-scanner contrast was a selection from them, and all three are
carried throughout here.

At the single pooled threshold **2.4766** (70% specificity on TUNE non-TB cases):

| scanner | n | TB | prevalence | AUROC | sensitivity | specificity | referral |
|---|---|---|---|---|---|---|---|
| 1 | 204 | 22 | 0.108 | 0.7013 | 0.545 | 0.764 | 0.270 |
| 2 | 665 | 75 | 0.113 | 0.7762 | 0.533 | 0.812 | 0.227 |
| 3 | 483 | 42 | **0.087** | 0.7290 | 0.809 | **0.524** | **0.505** |
| all | 1,352 | 139 | 0.103 | 0.7237 | 0.619 | 0.700 | 0.333 |

**Reconciliation with the previously reported table.** The earlier figures (rad_2 referral
24.6%, specificity 0.795, AUROC 0.7820; rad_3 referral 50.3%, specificity 0.520, AUROC
0.7000) were computed on the **full 2,708-case cohort**. The table above is **TUNE only**
(1,352 cases). The counts and definitions are otherwise identical; the differences are
entirely the change of denominator. Both are reproduced in
`../overnight_run_20261021_2340/baseline_by_scanner.csv` and here.

## 2. Thresholds, and how uncertain they are

Empirical 0.70 quantile of TUNE non-TB scores within each scanner
(`numpy.quantile`, linear interpolation):

| scanner | threshold | bootstrap 95% CI | width | n non-TB |
|---|---|---|---|---|
| 1 | 2.0070 | [1.5761, 2.4424] | 0.866 | 182 — **small subgroup** |
| 2 | 1.3887 | [1.0093, 1.7890] | 0.780 | 590 |
| 3 | 3.7075 | [3.3539, 4.0130] | 0.659 | 441 |
| pooled | 2.4766 | [2.2723, 2.6903] | 0.418 | 1,213 |

Every scanner threshold is estimated from a fraction of the data the pooled one uses, and
each interval is correspondingly wider — the rad_system 1 interval is more than twice the
pooled width on 182 negatives. **This is the main statistical cost of the method**, and it
scales badly: a scanner contributing few patients gets the least reliable threshold.

## 3. Results on REPORT — scored once, after the manifest was frozen

n = 1,356, 141 TB. Wilson intervals in `tables/threshold_report_metrics.csv`.

### Per scanner

| scanner | arm | sensitivity | specificity | referral | PPV | missed TB |
|---|---|---|---|---|---|---|
| 1 | A pooled | 0.591 | 0.747 | 0.289 | 0.220 | 9 |
| 1 | **B scanner** | **0.727** | 0.709 | 0.338 | 0.232 | **6** |
| 2 | A pooled | 0.605 | 0.778 | 0.265 | 0.260 | 30 |
| 2 | **B scanner** | **0.737** | 0.684 | 0.364 | 0.231 | **20** |
| 3 | A pooled | 0.674 | **0.516** | **0.501** | 0.119 | 14 |
| 3 | **B scanner** | **0.488** | **0.699** | **0.318** | 0.136 | **22** |

### Pooled, with paired uncertainty

4,000 paired bootstrap resamples on the identical REPORT cases:

| metric | A pooled | B scanner | difference | 95% CI | |
|---|---|---|---|---|---|
| sensitivity | 0.6241 | 0.6596 | **+0.0355** | [−0.0282, +0.0992] | includes 0 |
| specificity | 0.6782 | 0.6930 | +0.0148 | [−0.0041, +0.0338] | includes 0 |
| referral rate | 0.3532 | 0.3437 | −0.0096 | [−0.0273, +0.0081] | includes 0 |
| missed TB | 53 | **48** | −5 | | |

### Equity, which is where the method actually acts

| spread across the three scanners | A pooled | B scanner | change |
|---|---|---|---|
| specificity | **0.2625** | **0.0252** | **10× narrower** |
| referral rate | **0.2357** | **0.0468** | **5× narrower** |

---

## 4. Verdict

**Scanner-specific thresholds did not improve pooled operating-point metrics beyond
noise.** Sensitivity rose by 0.036 and specificity by 0.015, and every pooled confidence
interval includes zero. Five fewer tuberculosis cases were missed out of 141, which is well
within sampling variation.

**They did substantially equalise the burden across machines.** Specificity spread fell
from 0.26 to 0.03 and referral-rate spread from 0.24 to 0.05. That is the effect the method
is for, and it is large and consistent in direction.

**The cost falls on one group, and it is not small.** Scanner 3 sensitivity dropped from
0.674 to 0.488 — from 29 of 43 cases detected to 21 of 43. Scanner 3 patients were being
over-referred under the pooled threshold, and some of that over-referral was catching real
disease. Equalising specificity removes that accidental benefit. **Eight additional missed
cases on one machine, offset by thirteen fewer on the other two.**

So the honest summary is: **this trades detection on scanner 3 for detection on scanners 1
and 2, and buys a far more even workload.** Whether that is an improvement depends on
whether the programme values equal confirmatory-testing load or maximal detection for the
patients currently being over-referred. It is not a free win and must not be presented as
one.

## 5. Before anyone considers deploying this

- **Scanner identity must be known at inference.** Nothing in the current pipeline reads
  it. If it is inferred from the image, the inference becomes part of the model and
  inherits its own error rate, which this analysis has not measured.
- **Thresholds drift.** These were fitted on 182–590 negatives per machine at one point in
  time. Detector recalibration, a tube change or a software update invalidates them
  silently, because nothing in the output would look wrong.
- **An unseen fourth scanner has no threshold.** The method has no defined behaviour for a
  machine it was not fitted on, and the fallback would be the pooled threshold — exactly
  the behaviour being replaced.
- **Per-group thresholds are an equity intervention with an equity cost.** Deliberately
  applying a different decision rule by acquisition device needs a stated justification,
  and here it provably reduces detection for one group.
- **This cohort is exposed.** Vietnam TUNE chose the pooling configuration in earlier work,
  so REPORT is a controlled comparison, not an independent estimate. Everything above is
  **exploratory research, not a deployment recommendation.**
