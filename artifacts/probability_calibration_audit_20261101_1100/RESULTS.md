# Audit A1 — probability calibration of the existing outputs (Vietnam TUNE, exposed)

**Descriptive audit, no new fitting, no new lever. Research freeze (R1–R25) untouched.**
Protocol sha `49a7a0d7f2863212…` hashed before running. Cohort fingerprint pinned (1,352; 139 positive;
scanners 204 / 665 / 483). Reproduction gate: v3 raw TUNE AUROC 0.7585, v2 raw 0.7237 — both exact.
Head audited is the candidate file itself (`artifacts/p2/bundle_v3_candidate/head.npz`, sha `7f61a082…`),
checked equal to the R22 weights.

## Summary (10 equal-mass bins; 2,000 bootstrap draws, seed 20261101)

| output | AUROC | Brier (ref 0.0922) | Brier skill | ECE [95 % CI] | intercept | slope | reading |
|---|---|---|---|---|---|---|---|
| v2 image logit through a sigmoid | 0.7237 | 0.4254 | −3.61 | 0.487 [0.464, 0.510] | −5.27 | 0.30 | **not a probability** |
| v2 fused, out-of-fold | 0.7818 | 0.0838 | +0.091 | 0.024 [0.018, 0.044] | 0.00 | 0.94 | probability-shaped |
| v3 fused, out-of-fold | 0.7950 | 0.0821 | +0.110 | 0.016 [0.013, 0.035] | 0.00 | 0.94 | probability-shaped |
| comparator raw score | 0.7780 | 0.3313 | −2.59 | 0.439 [0.419, 0.459] | −3.41 | 0.68 | **not a probability** |

AUROCs here are of the repeat-averaged out-of-fold probability (0.7818 / 0.7950); R16/R22 reported the
mean of per-repeat AUROCs (0.7809 / 0.7941). Same quantity, different averaging order.

Per scanner (fused, out-of-fold): ECE 0.045 / 0.029 / 0.028 (v2) and 0.038 / 0.035 / 0.015 (v3);
intercepts within ±0.07 on every scanner. Scanner 1 (n = 204, 22 positive) is above the 0.03 bound in
both; the pooled point estimates pass, the pooled upper confidence limits (0.044, 0.035) do not.

## What this means

1. **The served screening score must not be read through a sigmoid.** It is a logit-adjusted,
   max-pooled regional output trained against healthy controls; the naive probability sits at 0.78 mean
   predicted against 0.10 observed in the top decile, intercept −5.3, slope 0.30. The API contract
   ("screening score, not a probability") is now a measured statement, not a disclaimer. The comparator's
   raw abnormality score is the same kind of object (intercept −3.4, slope 0.68) and should be treated
   the same way.
2. **The fused output is a probability for this cohort's prevalence and scanner mix — and only that.**
   Intercept 0.00 and slope 0.94 are what an unpenalised-ish logistic fit, out-of-fold on the same
   population, is expected to produce; the result says the 13-parameter fusion does not overfit (slope
   0.94, not 0.6) and that its probabilities would be honest *at a site with the same prevalence and the
   same three scanners*. It says nothing about any other site: the intercept is the TUNE prevalence and
   the scanner columns are TUNE's scanners (R23). At a new site the fusion is refitted or not used.
3. **Nothing changes in serving.** No temperature, Platt or isotonic map was fitted or saved
   (test `test_nothing_saved_for_serving`). The model card gains the numbers above; the API keeps
   returning the logit, and the fused probability remains an optional output that is only meaningful
   where its coefficients were fitted.

## Files

`tables/calibration_summary.csv`, `tables/calibration_per_scanner.csv`, `tables/reliability_bins.csv`,
`tables/run_manifest.json`; `scripts/calib_audit.py`; `tests/test_calib_audit.py` (6 pass).
