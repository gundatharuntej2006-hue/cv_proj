# Protocol — clinical-context fusion, nested cross-validation on Vietnam TUNE

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory.** TUNE is exposed (it selected the pooling configuration), so the *absolute*
AUROCs here are optimistic. The *increment* from fusing clinical context — fitted and
scored entirely inside cross-validation folds — is the quantity of interest. **No result
here is validation, and nothing is adopted into production by this run.**

## 1. Question

> Does combining the frozen image score with the clinical fields a screening form already
> collects improve discrimination — overall and in the previous-TB population — over the
> image score alone? And does the same fusion help a commercial image-only comparator as
> much, i.e. is any gain specific to our score or generic?

## 2. Cohort

Vietnam TUNE, 1,352 participants, 139 Xpert-positive, 466 previous TB, fingerprint
`1aa55cd5c0eb2088…` (asserted). **REPORT not read. `internal_test` not referenced.**

## 3. Inputs

| image score | source |
|---|---|
| `ours` | production score ≡ arm A, `active_tb_label_raydino_20261024_1254/.../vietnam_tune_armA.csv` |
| `ours_B1` | arm B1 (active-TB target), same directory — secondary |
| `lunit` | `lunit_3901` from the Dryad clinical CSV — comparator |

**Clinical features (12)**, all from the Dryad CSV, none derived from images:
`sex`, `age_group` (recoded 1/2 → 0/1), `c` cough, `f` fever, `wl` weight loss, `ns` night
sweats, `contact`, `past_tb`, `dm` (0/1, with `dm = 99` → 0 plus a `dm_unknown` indicator),
`rad_system` as two indicators (scanner 1, scanner 2; scanner 3 reference).

## 4. Models — one fixed specification, no hyperparameter search

Logistic regression, L2, **C = 1.0**, `lbfgs`, max 1000 iterations; the image score is
standardised on the training fold; binary features enter as 0/1. No tuning of C, no
feature selection, no class weighting.

| model | inputs |
|---|---|
| **M0** | image score alone (no fitting — the score's own AUROC) |
| M1 | clinical features alone |
| **M2** | image score + clinical features |
| M3 | M2 + `image_score × past_tb` interaction (a previous-TB-specific slope) |

Each is run for each image score (`ours`, `ours_B1`, `lunit`).

## 5. Cross-validation

Stratified **10-fold**, **repeated 20 times** (seeds 0–19). For each repeat, the
out-of-fold predictions over all 1,352 participants are pooled and scored once. Reported:
mean and SD across the 20 repeats, and a **paired bootstrap** (2,000 resamples of
participants, seed 20261101) on the repeat-averaged out-of-fold predictions for the
contrasts M2−M0, M3−M0, M3−M2, and `ours`-fused − `lunit`-raw, `ours`-fused −
`lunit`-fused.

Operating point: within each fold, the threshold reaching ≥ 0.70 specificity on the
**training** portion's fitted probabilities is applied to the held-out fold; pooled
sensitivity and specificity are reported. (Training-fold thresholds are mildly optimistic
for specificity; this is stated, not hidden.)

## 6. Endpoints

**Primary:** paired ΔAUROC, M2 − M0, `ours`, all 1,352. **Minimum meaningful effect
+0.02**, interval excluding zero — the same bar this project used for screening-cohort
interventions.

**Secondary:** AUROC in the previous-TB subgroup (n = 466; 49 positive) and the
no-previous-TB subgroup (n = 886); sensitivity/specificity at the operating point; the
same for M3; the comparator track; AUPRC with prevalence.

## 7. Interpretation rules, fixed now

- A gain for `ours` that the comparator track matches is a **generic** clinical-context
  effect — useful for a product, not evidence about our image model.
- A gain in the previous-TB subgroup is reported with its n = 49 positives stated
  prominently; it cannot be precise.
- No sensitivity gain is reported without its specificity.
- Nothing here is adopted. A positive result licenses *building the fusion into the
  serving layer as an optional input*, nothing more.
