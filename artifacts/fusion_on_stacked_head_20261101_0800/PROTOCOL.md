# Protocol — clinical-context fusion on the MAIRA-2 + active-target head

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory, and more exposed than R16.** Everything here is on Vietnam TUNE
(1,352; Xpert reference), by nested cross-validation, on a head whose encoder and
target were both chosen on TUNE (R21). Nothing is adopted into production.

## 1. Question

R16 measured fusion on the production head (A): 0.7237 → 0.7809 (+0.058), parity with
the comparator's raw score, and the comparator gains nothing from the same fusion.
R21 found a head that is +0.035 better than A on TUNE before fusion (X_noNLM: MAIRA-2
encoder, active-TB target, TBX11K training only; 0.7585).

> **Does the head-level gain survive fusion?** If the two gains are even partly
> additive the fused X_noNLM score exceeds the comparator's raw score on this cohort.

## 2. Arms

Image scores, each seed-mean over seeds 17 / 29 / 43 / 61 / 79:

| key | head | gate |
|---|---|---|
| `ours` | A — production (R16 input, `vietnam_tune_armA.csv`) | M2 must reproduce 0.7809 |
| `ours_X` | X_noNLM — refitted here with the R20 trainer on the R20 MAIRA-2 caches | raw TUNE AUROC must reproduce 0.7585 and calibration 0.9466 (R21) |
| `lunit` | comparator, `lunit_3901` column | M2 must reproduce 0.7778 |

Fusion models exactly as R16: **M0** raw score; **M2** logistic regression (C = 1, L2)
on the fold-standardised score + the 12 clinical features (sex, age_group, c, f, wl,
ns, contact, past_tb, dm, dm_unknown, scan1, scan2). 10-fold stratified × 20 repeats,
out-of-fold predictions averaged over repeats. No new feature, no interaction, no
re-tuning. M1 and M3 are not rerun (R16 settled them).

## 3. Evaluation

**Primary:** paired ΔAUROC on TUNE, repeat-averaged OOF, **`ours_X` M2 minus `ours`
M2** — does the head gain survive fusion. 2,000 paired resamples, seed 20261101 (R16).
Minimum meaningful effect **+0.02**, interval excluding zero.

**Secondary:** `ours_X` M2 − `lunit` M0 (fused ours vs raw comparator); `ours_X` M2 −
`lunit` M2 (both fused); `ours_X` M0 − `ours` M0 (reproduces R21's +0.0348); each
contrast in the previous-TB subgroup (n = 466); sensitivity / specificity at the
within-fold 0.70-specificity threshold; AUPRC.

**Decision rule.** Nothing is adopted on any outcome. A positive secondary against the
comparator is a *product-direction* finding on an exposed cohort and is written as
such.

## 4. Partitions

TBX11K `internal_test` (901) and Vietnam REPORT (1,356) are not read. The TUNE cohort
fingerprint must equal the pinned `1aa55cd5…`. No per-case output; no patient
identifier in any table.
