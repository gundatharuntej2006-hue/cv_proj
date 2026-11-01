# Phase 3 — connecting the label-alignment result to the CLEAR concept findings

**The planned analysis could not be run. The dependency is missing, and CLEAR was
deliberately not re-run to create it.** What follows uses the aggregate CLEAR results
only, and is reasoning, not measurement.

---

## 1. The missing dependency, stated exactly

The planned test was: does the score–concept association change between arm A and arm B1?
That needs **per-case** concept similarities joined to **per-case** classifier scores.

| ingredient | available? | where |
|---|---|---|
| per-case classifier scores, arms A / B1 / B2, Vietnam TUNE | **yes**, written by this run | `outputs/predictions/vietnam_tune_arm{A,B1,B2}.csv` |
| per-case classifier scores, arms A / B1 / B2, TBX11K calibration | **yes**, written by this run | `outputs/predictions/calibration_arm{A,B1,B2}.csv` |
| **per-case CLEAR concept similarities** | **NO** | not on disk anywhere |
| per-case CLEAR similarities for **TBX11K** | **NO, and never computed** | CLEAR was run on Vietnam TUNE only |

The concept audit exported four aggregate tables and nothing else —
`artifacts/concept_audit_result_20261024_0130/outputs/tables/` holds 37, 10, 37 and 10
rows. The 1,352 x 9 per-case matrix was **deliberately** not exported, to keep patient-level
data off the Kaggle round trip. That decision was right and is not being second-guessed.

Per the task constraints, CLEAR was **not** re-run to manufacture the missing table.
Re-running it is a one-Kaggle-job option, listed in §4.

## 2. What the aggregates still support — and it is not nothing

| CLEAR finding (Vietnam TUNE, 1,213 TB-negatives) | value |
|---|---|
| scarring similarity vs classifier score | rho = +0.566 [0.528, 0.602], delta-R2 = 0.198 |
| fibrosis similarity vs classifier score | rho = +0.417, delta-R2 = 0.100 |
| scarring similarity vs **Xpert status** | AUROC **0.494** [0.445, 0.542] — chance |

Read together: a chronic-appearance axis explains a large share of the classifier's
variance among people who do not have TB, while carrying no information about who does.
That is the signature of a nuisance direction, and it is what motivated testing the label
change.

**This run's result bears on that, from a different angle.** If the latent-TB training
label were the main thing pushing the classifier along that axis, relabelling it should
move the score. It does — but by very little, and only where it is cheap:

| contrast (TBX11K calibration, POST-HOC) | arm A | arm B1 |
|---|---|---|
| active TB vs `latent_tb` (n = 81 vs 18) | 0.7414 | **0.7524** |
| active TB vs `sick_non_tb` (n = 81 vs 400) | 0.9114 | **0.9035** |

The latent contrast moves in the predicted direction and the non-TB-abnormal contrast
moves against it, and the two roughly cancel. On the external screening cohort the same
change is worth **+0.0203** AUROC (exploratory, Vietnam TUNE).

**So the label is part of the story, not the whole of it.** A mechanism worth about +0.02
on a screening cohort cannot be the mechanism behind an association with delta-R2 = 0.198.
Most of the chronic-appearance signal survives the label change.

## 3. What may NOT be concluded from this

- **Not causality.** Neither the correlation nor its persistence shows the classifier uses
  a visual scarring cue. Both are consistent with a third factor driving score and
  similarity together.
- **Not "the visual mechanism was removed".** Nothing was removed; a training label was
  changed. No association was re-measured, because the per-case data to re-measure it
  does not exist.
- **Not a clinical label.** A CLEAR similarity is a weakly supervised text-similarity
  proxy against one report sentence. It is not a radiologist annotation and may not be
  treated as one.
- **Not a basis for subtraction.** No concept score was used to adjust, subtract from or
  suppress any TB score here, and none may be.
- **Not transferable to TBX11K.** CLEAR has never been run on TBX11K, so no concept
  statement in this document applies to the calibration result directly.

## 4. What would actually answer the question, in order of cost

1. **Re-run the TUNE concept audit exporting a 9x9 concept correlation matrix and a
   negative-control concept** (both aggregates, no per-case export). This settles whether
   the nine axes are near-collinear and whether an unrelated concept also tracks the
   score — open items U2 and U4 of the previous audit. One Kaggle job, touches nothing
   reserved. **Highest value per unit of effort.**
2. **Re-run it a second time over arm B1's scores** instead of the production scores, and
   compare the two aggregate rho values. This is the Phase 3 analysis as designed. It
   needs the arm B1 Vietnam TUNE scores, which this run has now produced. Still TUNE only.
3. **Resolve the consolidation anomaly** (rho = -0.300 among negatives, -0.385 among
   positives). Until it is understood, the nine axes cannot be read as radiological
   findings, and (1) and (2) are measurements of something unnamed.

Nothing in this list requires Vietnam REPORT or TBX11K `internal_test`.
