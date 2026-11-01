# Previous-TB subgroup: verified analysis

**Run:** `artifacts/prevtb_subgroup_20261021_1055/` · 2026-10-21 · HEAD `f878a6c`, unchanged
**Nothing was trained, re-scored, recalibrated or re-thresholded. TBX11K `internal_test`
was not opened. No model improvement is claimed.**

---

## Headline

**The subgroup difference is statistically supported.** Our model discriminates
substantially worse among participants with a documented history of previous TB, and the
degradation is significantly larger than a commercial comparator's on the identical
participants.

| | estimate | 95% CI | bootstrap p |
|---|---|---|---|
| **our degradation** (no-previous-TB minus previous-TB) | **0.1629** | **[0.1372, 0.1890]** | < 0.0001 |
| comparator degradation (Lunit 3.9.0.1) | 0.0978 | [0.0770, 0.1196] | < 0.0001 |
| **difference in differences (ours − Lunit)** | **+0.0651** | **[0.0399, 0.0900]** | < 0.0001 |

**But the subgroup was not prespecified**, the cohort is exposed, and causation is not
established. Those limits are in §6 and they are not incidental.

## 1. Reproduction of the reported figures

All seven checks pass at tolerance 0.0001 (`tables/metric_reproduction.json`).

| quantity | reported | reproduced | |
|---|---|---|---|
| ours, negatives without previous TB | 0.7811 | **0.7811** | pass |
| ours, negatives with previous TB | 0.6182 | **0.6182** | pass |
| Lunit, without previous TB | 0.8170 | **0.8170** | pass |
| Lunit, with previous TB | 0.7192 | **0.7192** | pass |
| n = 1,858 / 1,130, TB = 280 | — | identical | pass |

**Source traced.** The reported figures come from
`artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv`
(`score_baseline`), **not** from the Phase 2 experiment file. That matters — see §5.

## 2. The `past_tb` field

| | |
|---|---|
| definition | *"past_tb (past history of TB): 0/No; 1/Yes"* — `data/raw/vietnam/README.md`, Dryad DOI 10.5061/dryad.5tb2rbpcv |
| values present | 0 → 1,756 · 1 → 952. **No blanks** |
| explicit unknown code | **none** — unlike `dm`, which carries a `99` "Don't Know or Refused" |
| verification | self-reported; the README does not state whether records were checked |

**Unresolved and marked unverified:** because there is no unknown code, it cannot be
determined from the released data whether histories were obtained from every participant
or whether unknown histories were coded `0`. If the latter, the "no previous TB" stratum is
contaminated with unknowns, which would **attenuate** the measured gap — the true effect
would be larger, not smaller. The direction of that bias is knowable; its size is not.

## 3. Subgroup composition

| | n | TB | negatives | prevalence |
|---|---|---|---|---|
| negatives **without** previous TB + all positives | 1,858 | 280 | 1,578 | 0.1507 |
| negatives **with** previous TB + all positives | 1,130 | 280 | 850 | 0.2478 |
| whole cohort | 2,708 | 280 | 2,428 | 0.1034 |

**No exclusions.** The two subgroups partition the negatives and **share every positive**.

Previous TB is present in **35.0%** of negatives and **36.4%** of positives (102/280), so
it is common in both classes.

**The prevalence differs between the subgroups by construction**, because the shared
positives form a larger share of the smaller subgroup. This does not bias the comparison:
AUROC is invariant to class balance when the class-conditional score distributions are
fixed.

## 4. Why a naive two-sample test would have been wrong, and what was done instead

**The two subgroup AUROCs share all 280 positives.** They are not independent samples, so
comparing their confidence intervals, or applying a two-sample test, would be invalid.

**Method used:** a stratified bootstrap, 10,000 replicates, seed 20261021. Within each
replicate the positives are resampled **once** and that same positive resample is used for
both subgroup AUROCs; each negative stratum is resampled independently. The difference is
computed within the replicate, so the shared-positive dependence is carried through.

**Patient-level resampling.** Vietnam is one image per patient — 2,708 unique `pid` for
2,708 rows, asserted in code — so patient-level and case-level resampling coincide and no
clustering correction is needed. **This is a property of this cohort, not a general one.**

Per-subgroup intervals (same bootstrap):

| model | subgroup | AUROC | 95% CI | AUPRC |
|---|---|---|---|---|
| ours | without previous TB | 0.7811 | [0.7553, 0.8053] | 0.4169 |
| ours | with previous TB | 0.6182 | [0.5812, 0.6544] | 0.3430 |
| Lunit | without previous TB | 0.8170 | [0.7901, 0.8420] | 0.4997 |
| Lunit | with previous TB | 0.7192 | [0.6816, 0.7545] | 0.4607 |

**A supporting statistic with no shared-sample issue.** Among negatives only, our score
separates previous-TB from non-previous-TB participants at AUROC **0.6914
[0.6698, 0.7126]**. The model is, in effect, partly detecting treatment history.

## 5. A reproducibility finding worth recording

Arm H of the Phase 2 experiment is the **same configuration refitted from scratch**
(regional 12×12 max, lungs-only, healthy-only negatives, seeds 17/29/43/61/79). Comparing
the two fitted heads' scores on the same 2,708 participants:

| | |
|---|---|
| scores identical | **no** |
| mean absolute difference | **0.849 logits** |
| max absolute difference | 2.335 logits |
| Spearman rank correlation | **0.9979** |
| AUROC, without previous TB | 0.7822 against 0.7811 |
| AUROC, with previous TB | 0.6173 against 0.6182 |
| AUROC, full cohort | 0.7245 against 0.7240 |

**Head training is not bit-reproducible. Rankings reproduce to ~0.001 AUROC; the absolute
score scale does not.** Two consequences:

1. The ±0.001 spread between the two fits is **two orders of magnitude smaller** than the
   0.163 subgroup gap, so the finding is unaffected.
2. **Any per-source offset or threshold fitted against one fitted head does not transfer
   to another refit of the same configuration.** That is a direct, previously unrecorded
   argument against the scanner-offset approach explored in an earlier run, independent of
   whether that approach works.

## 6. Interpretation, stated cautiously

**Supported.** Our model ranks active TB against previous-TB negatives substantially worse
than against other negatives — a gap of 0.163 [0.137, 0.189] in AUROC. Because this is a
threshold-free ranking statistic, **a per-group threshold or recalibration cannot repair
it.** The commercial comparator degrades on the same axis but less, and the
difference-in-differences interval excludes zero.

**Plausible but not demonstrated.** Residual radiographic changes from treated disease —
fibrosis, cavities, volume loss, calcification — resemble active TB, and a model trained on
`active_tb` plus `latent_tb` against healthy-only negatives has no way to learn the
distinction. **This experiment does not test that mechanism.** It measures an association
between a self-reported history variable and model ranking.

**Explicitly not claimed.**

- Not causal. Previous TB is self-reported, correlates with age, prior care-seeking,
  symptoms and comorbidity, and no mediation or adjustment analysis was performed.
- Not an explanation of the whole screening gap. The system reaches 0.7811 even among
  negatives **without** previous TB, against the commercial comparator's 0.8170 — a deficit
  of 0.036 that has nothing to do with this subgroup.
- Not prespecified. See below.
- Not validation. Vietnam informed the pooling choice and its REPORT half has been scored
  in several runs.

**Multiplicity and selection.** The subgroup was **discovered during exploratory
inspection** in run `realworld_generalization_20261021_1015`, where five clinical variables
were screened (`past_tb`, `dm`, symptom count, `contact`, `age_group`). The intervals above
are **not corrected** for that selection. The effect is large enough to survive a
Bonferroni correction across five variables — the bootstrap p is below the 1/10,000
resolution of the procedure — but the point stands that this is a hypothesis generated and
tested on the same data.

**Reference-standard caveat.** Xpert Ultra can remain positive, or return "trace", in
previously treated patients. That would create false *positives* in the previous-TB
stratum, which would push the measured gap in the **opposite** direction to the observed
one — so it does not explain the finding, but it does muddy the reference standard in
precisely this subgroup. Whether trace results were adjudicated is not stated in the Dryad
documentation. **Unverified.**

## 7. Reproduction

```
py -3 artifacts/prevtb_subgroup_20261021_1055/scripts/verify_subgroup.py
py -3 artifacts/prevtb_subgroup_20261021_1055/scripts/power_for_subgroup_study.py
py -3 -m pytest artifacts/prevtb_subgroup_20261021_1055/tests -q
```

Runtime about 90 seconds, CPU only, no GPU required. Inputs:
`overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv` and
`data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv`; both hashed in
`tables/metric_reproduction.json`.
