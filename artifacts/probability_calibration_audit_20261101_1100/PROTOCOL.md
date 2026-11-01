# Audit A1 — are any of the served or candidate outputs usable as probabilities?

**Type:** descriptive audit, no fitting of anything new, no new lever. It does not reopen the
research freeze (R1–R25): every quantity below is a property of an output that already exists.
**Date:** 2026-11-01. **Owner:** P2.

## Question
The serving contract says the TB screening score is a logit, not a probability. That statement has
never been measured. This audit measures it, for the outputs a consumer could be tempted to read as
probabilities.

## Cohort
Vietnam TUNE, 1,352 participants, 139 Xpert-positive (exposed cohort; fingerprint pinned, checked by
`fusion_cv.load`). REPORT and TBX11K `internal_test` are not read.

## Outputs audited (all pre-existing)
1. `v2_raw_sigmoid` — sigmoid of the served v2 image logit (arm A seed-mean), as a naive consumer
   would read it.
2. `v2_fused_oof` — the v2 fusion (R16 model M2) out-of-fold probability, mean over the 20 repeats
   of 10-fold nested CV, exactly as R16 computed it.
3. `v3_fused_oof` — same for the v3 candidate (R22 head `7f61a082…`, scores recomputed from the R20
   MAIRA-2 cache and the saved head; no retraining).
4. `lunit_raw` — the comparator's abnormality score on its own [0, 1] scale.

## Metrics (fixed before running)
- Brier score; Brier of the prevalence-only reference; Brier skill score.
- Expected calibration error, 10 equal-mass bins.
- Calibration intercept (logit offset at slope 1) and calibration slope (Cox), fitted by
  unpenalised logistic regression of the outcome on logit(p). These two fits are the audit's
  instruments, not models; nothing from them is saved for serving.
- Reliability table, 10 equal-mass bins.
- 2,000 bootstrap draws, seed 20261101, for Brier and ECE.
- The fused outputs' ECE and intercept per scanner.

## Pre-specified reading
An output is **"probability-shaped"** on this cohort if slope in [0.8, 1.2], intercept in
[−0.2, 0.2] and ECE ≤ 0.03 at once. Anything else is **"not a probability"**.
Whatever the reading, the served contract does not change: a probability-shaped out-of-fold result on
an exposed cohort licenses a sentence in the model card, not a change in what the API returns.

## Not done here
No temperature, Platt or isotonic recalibration of any served output is fitted or saved.
