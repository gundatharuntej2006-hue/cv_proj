# Phase 3 — Sample size and power

Computed in `scripts/power_planning_note.md` terms from committed per-case scores by
resampling (`tables/power_recalculation.json`). **Recalculated for this run's endpoint; not
copied from the earlier estimate.**

---

## 1. Method

Empirical SE of the statistic from 4,000 paired bootstrap resamples of the actual Vietnam
per-case scores, extrapolated as SE proportional to 1/sqrt(n), then inverted:

    n = n0 * ((z_alpha + z_beta) * SE0 / delta)^2

Two-sided alpha 0.05, power 0.80 (z = 1.959964, 0.841621).

## 2. Assumptions, all of which must be re-checked against the real cohort

- case mix, negative-class spectrum and score distributions resembling the Vietnam cohort;
- **past TB in 35.0% of negatives**, as observed in Vietnam;
- one image per patient, so cases are independent;
- paired design: both arms scored on identical patients;
- SE scales as 1/sqrt(n); approximate normality of the statistic;
- **no multiplicity adjustment.** Phase 2 examined three endpoints; a real study should
  adjust or declare a single primary endpoint.

## 3. The observed quantities these are built from

Primary endpoint: AUROC, active TB vs Xpert-negative participants with past TB.
n = 1,130 (280 TB, 850 negatives).

| quantity | value |
|---|---|
| arm H AUROC | 0.6173 |
| arm L AUROC | 0.6465 |
| paired delta (L - H) | +0.0292 |
| **SE of the paired delta** | **0.00355** |
| **SE of a single-arm AUROC** | **0.01912** |

## 4. The design insight that follows

**The paired SE is 5.4x smaller than the single-arm SE.** Two arms sharing a backbone,
features and seeds differ only in the label mapping, so their scores are highly correlated
and their difference is measured very precisely. The consequence is counter-intuitive and
decides the study design:

| goal | required subgroup n | required TB cases |
|---|---|---|
| detect a paired delta of 0.029 | **134** | 33 |
| detect a paired delta of 0.05 | 45 | 12 |
| detect a paired delta of 0.075 | 20 | 5 |

| goal | required subgroup n |
|---|---|
| estimate one arm's AUROC to +/- 0.05 | 635 |
| estimate one arm's AUROC to +/- 0.04 | 992 |
| **estimate one arm's AUROC to +/- 0.03** | **1,763** |
| estimate one arm's AUROC to +/- 0.02 | 3,967 |

**Comparing two of our own arms is cheap. Knowing how good either one actually is, is
expensive.** The new cohort is needed for the second purpose, not the first.

## 5. Translating to participants screened

The primary subgroup is {active TB} + {Xpert-negative with past TB}. At Vietnam's observed
rates — prevalence 0.103, past TB in 35.0% of negatives — each 1,000 participants screened
yields roughly:

    103 TB + 0.35 * 897 = 103 + 314 = ~417 subgroup cases

| target | subgroup n | **participants to screen** |
|---|---|---|
| AUROC to +/- 0.05 | 635 | **~1,500** |
| AUROC to +/- 0.04 | 992 | **~2,400** |
| **AUROC to +/- 0.03** | **1,763** | **~4,200** |

**Recommended target: 2,500-4,000 participants** with universal microbiological
confirmation. That estimates the past-TB subgroup AUROC to roughly +/- 0.035-0.04, is
comfortably powered for any paired arm comparison, and leaves room for the missingness in
section 6.

Below ~1,500 participants the study can compare arms but cannot say how good the system
is, which is the question that matters.

## 6. Realistic missingness, and what it costs

| loss | plausible rate | effect |
|---|---|---|
| no sputum produced / no Xpert result | 5-15% | removes the case entirely under universal verification |
| unreadable or non-frontal image | 1-3% | as above |
| past-TB history missing or refused | 5-10% | removes the case from the **primary subgroup only** |
| segmentation failure | ~0.6% (observed on DA/DB) | small |

Compounded, plan for **15-25% attrition from the primary subgroup**. Inflate the target
accordingly: a 4,200-participant plan should be budgeted at **~5,000 enrolled**.

## 7. Sensitivity analyses

**If past TB is rarer than 35%.** Zambia (Kagujje et al. 2022) reported 24.0%. At 24% the
subgroup yield per 1,000 falls from ~417 to ~319, so **participants required rise by about
30%**: the +/- 0.03 target moves from ~4,200 to ~5,500.

**If prevalence is lower than 10.3%.** At 5% prevalence and 35% past TB, yield per 1,000 is
50 + 333 = 383 - only ~8% worse, because the subgroup is dominated by negatives. But the
number of **positives** falls by half, which widens every sensitivity estimate. Prevalence
matters far more for sensitivity endpoints than for this AUROC endpoint.

**If the effect is smaller than observed.** The +0.029 estimate comes from an exposed
cohort and may be optimistic. At a true +0.015 the paired requirement rises to ~500
subgroup cases - still far below the precision requirement, so it does not bind.

**If scores are more variable in a new setting.** A 50% larger SE multiplies every n by
2.25. The +/- 0.03 precision target would then need ~9,500 participants. This is the single
largest risk to the plan and argues for reporting precision achieved rather than promising
it in advance.

## 8. What these numbers are not

They size a study to **estimate** performance and to **compare two of our own arms**. They
do not size a study to show non-inferiority to a commercial product, to demonstrate the WHO
target product profile is met, or to support regulatory submission - each of which has
different and generally larger requirements.
