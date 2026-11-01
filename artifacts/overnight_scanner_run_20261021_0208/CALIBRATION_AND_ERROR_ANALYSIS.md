# Task 5 — Offset or ranking? Where the scanner disparity actually lives

**All calibrators and offsets fitted on TUNE. The REPORT figures in §4 come from the single
sealed pass after `FROZEN_MANIFEST.md` was written.**

Tables: `tables/calibration_by_scanner.csv`, `tables/offset_vs_ranking.json`,
`tables/subspace_metrics.csv`

---

## 1. The two hypotheses, and how to tell them apart

A per-scanner disparity at a fixed threshold can come from two different failures:

- **Ranking** — the model orders patients worse on one machine. Shows up as a lower
  per-scanner AUROC. Cannot be fixed by moving a threshold.
- **Offset** — the model orders patients fine but puts the whole machine's scores on a
  different scale. Shows up as a shifted score distribution at equal AUROC. Can be fixed
  without touching the model.

They are distinguishable because **AUROC within a group is invariant to adding a constant
to that group's scores**. So subtracting each scanner's own score offset leaves every
per-scanner AUROC exactly unchanged, and changes only how the groups interleave when
pooled.

## 2. Ranking varies modestly; the offset is enormous

TUNE, n = 1,352:

| scanner | AUROC | 95% CI | median score, non-TB | median score, TB | sd, non-TB |
|---|---|---|---|---|---|
| 1 | 0.7013 | [0.60, 0.80] | **+0.514** | 2.943 | 2.19 |
| 2 | 0.7762 | [0.72, 0.83] | **−0.861** | 2.619 | 2.23 |
| 3 | 0.7290 | [0.65, 0.81] | **+2.345** | 4.420 | 2.14 |

AUROC spans **0.075**. The median healthy person spans **3.21 logits** between scanner 2
and scanner 3 — larger than the within-group standard deviation, and larger than the gap
between healthy and diseased patients on scanner 2 itself (−0.861 to 2.619 = 3.48).

**Calibration quality confirms it.** With a single pooled Platt calibrator the calibration
slope is 0.887 on scanner 1, 1.259 on scanner 2 and 1.251 on scanner 3; perfect is 1.0.
Fitting a calibrator per scanner lowers the Brier score on every machine (0.0926 → 0.0922,
0.0925 → 0.0907, 0.0783 → 0.0758). The improvements are small because Brier is dominated by
the 10% prevalence, but they run the same way on all three.

## 3. Quantifying the offset's cost — TUNE

Subtracting each scanner's TUNE non-TB median:

| | before | after |
|---|---|---|
| pooled AUROC | 0.7237 | **0.7533** |
| referral-rate spread across scanners | 0.278 | 0.124 |

**+0.0296 pooled AUROC, purely from re-aligning the three score scales.** Since
within-scanner AUROC cannot change under a constant shift, every point of that gain comes
from fixing how the machines interleave when pooled. Put plainly: **about 0.03 AUROC of the
system's pooled performance was being destroyed by the scanner offset alone.**

That is larger than anything the subspace experiment produced (0.0002) and comparable to
the entire measured benefit of focal pooling over the global representation on the curated
sites (0.028–0.040).

## 4. REPORT — the honest version

The same offsets, estimated on TUNE, applied unchanged to REPORT in the sealed pass:

| arm | pooled AUROC | Δ vs baseline | 95% CI | |
|---|---|---|---|---|
| baseline | 0.7242 | — | | |
| **offset-corrected** | **0.7448** | **+0.0206** | [−0.0030, +0.0436] | **includes 0** |

Per-scanner AUROC changed by exactly 0.0000 on all three machines, which is the arithmetic
guarantee working as it should and a useful internal check that the correction was applied
correctly.

**The TUNE gain of +0.0296 shrank to +0.0206 on REPORT, and the interval includes zero.**
The direction is consistent across both halves and the point estimate is substantial, but
**this is not a statistically established improvement** and must not be reported as one.
With 1,356 cases and 141 positives there is not enough power to resolve an effect of this
size; the interval spans −0.003 to +0.044.

Note also that the offset correction is **not** part of the preregistration — it arose from
the TUNE calibration analysis during this run. It is exploratory on both counts.

## 5. Error profile at the operating point

From `THRESHOLD_RESULTS.md`, REPORT, at the frozen thresholds:

| | pooled threshold | scanner thresholds | offset-corrected, single threshold |
|---|---|---|---|
| pooled sensitivity | 0.624 | 0.660 | 0.638 |
| pooled specificity | 0.678 | 0.693 | 0.691 |
| pooled referral rate | 0.353 | 0.344 | 0.343 |
| missed TB (of 141) | 53 | **48** | 51 |
| specificity spread across scanners | **0.263** | **0.025** | 0.093 |

The three approaches land in nearly the same place on pooled metrics and differ mainly in
how evenly the burden falls. Scanner-specific thresholds equalise it best; the offset
correction achieves most of the equalisation with a *single* threshold, which is the more
defensible object to deploy — but both depend on knowing the scanner.

## 6. The trade-off, stated plainly

Aligning the scales does not create detections; it moves them.

- Scanner 3 was being over-referred (50.1% at the pooled threshold, at the **lowest**
  prevalence of the three). Correcting that reduces its sensitivity from 0.674 to 0.488 —
  **eight more missed cases out of 43**.
- Scanners 1 and 2 were being under-referred. Correcting that raises their sensitivity from
  0.591 to 0.727 and 0.605 to 0.737 — **thirteen fewer missed cases out of 98**.

Net across the cohort: five fewer missed cases and a slightly smaller referral burden. But
**the disparity is not "fixed"**, it is redistributed, and one group is measurably worse
off. A programme that currently relies on scanner 3's accidental over-referral would lose
cases under any of these corrections.

## 7. Caveats

- All offsets and thresholds come from 182–590 negatives per machine on a single cohort.
- Vietnam TUNE chose the pooling configuration in earlier work, so REPORT is a controlled
  comparison, not independent validation.
- Both corrections require the scanner to be known at inference time, which the current
  pipeline does not read, and neither has defined behaviour for an unseen machine.
- Nothing here is prospective or clinically validated.
