# Preregistered analysis manifest — B1 label alignment x CLEAR mechanism test

**Written and hashed before any concept inference was run or any association examined.**
Hash in `tables/manifest_hash.txt`; the run script re-hashes this file and refuses to
start if it has changed.

**Classification: exploratory mechanism analysis on an exposed cohort.** Not independent
validation. Nothing here may be read as a confirmed model improvement, as evidence of
clinical benefit, or as proof that the classifier causally uses a visual cue.

---

## 1. Question

> Is the B1 (active-TB target) AUROC change on Vietnam TUNE accompanied by a measurable
> reduction in the association between the TB classifier score and CLEAR's scarring
> concept similarity among TB-negative cases, **without** evidence of harmful score
> changes among TB-positive cases?

The answer is not presumed. "No change", "the control moves too" and "harm" are all
outcomes.

## 2. Arms — and why B0 is the refit, not the production score file

Three score columns over the **same 1,352 TUNE cases, same order, same labels**:

| key | source | role |
|---|---|---|
| `baseline_production` | `score_baseline` in `overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv` | **reproduction gate only** (§7). The exact column the 2026-10-23 Kaggle audit used |
| **B0** | `active_tb_label_raydino_20261024_1254/outputs/predictions/vietnam_tune_armA.csv` | the comparator |
| **B1** | `.../vietnam_tune_armB1.csv` | the active-TB-target arm |

**B0 is arm A, not `score_baseline`.** Arm A is a refit of the production configuration
through the identical fitting path as B1 — same frozen RAD-DINO features, same regional
12x12 max head, same optimiser, same five seeds — differing from B1 **only** in the
training target. Pairing B1 against `score_baseline` would confound the target change with
the refit. `score_baseline` is carried anyway, purely so that this run's CLEAR pipeline can
be checked against the published Kaggle numbers.

## 3. Concepts

**Primary family: the 9 pinned concepts**, unchanged from `configs/concept_indices.json`
of `kaggle_exec_prep_20261023_1130` — fibrosis (835), scarring (96952), volume loss
(368164), calcified granuloma (209383), calcification (77874), focal opacity (281312),
consolidation (123742), pleural thickening (341471), bronchiectasis (57870). Rule
unchanged: **exact single vocabulary entry only**; cavity, nodule and reticular opacity
remain excluded and are **not** approximated.

### Negative controls — declared now, before inference

| control | vocabulary entry | index | exact? |
|---|---|---|---|
| **primary control** | `scoliosis` | **82734** | yes — one entry |
| secondary control | `degenerative changes` | **339316** | yes — one entry |

Verified against `mimic_concepts.csv`, sha-256
`93e975b722422b64f18b57161ac9894fe30ac7a04cc68b7aa8a2e01f6d0fc439` — the same checksum
`concept_indices.json` was mapped against. The four pinned indices that could be
re-derived (scarring, fibrosis, consolidation, volume loss) all reproduce exactly, which
confirms the vocabulary file is the one the earlier mapping used.

**Rationale, recorded before running.** A negative control has to be visible on a frontal
chest radiograph, have real variance in a rural adult screening population, and have no
mechanistic link either to active TB or to post-TB chronic parenchymal change. `scoliosis`
satisfies all three: it is skeletal, common, and unrelated to pulmonary parenchyma.
Device-type entries (`pacemaker`, `sternotomy wires`, `surgical clips`) are exact entries
too but were **rejected** — in this cohort they would be near-absent, giving a degenerate
control that cannot move and therefore cannot fail.

`degenerative changes` is a second, weaker control: also skeletal and common, but it
correlates with age, which is already a covariate. It is reported so that one control
cannot carry the argument alone. **It is not a substitute for `scoliosis`.**

Controls are analysed through the identical pipeline but are **excluded from the
Benjamini-Hochberg family** — they are controls, not hypotheses.

## 4. Covariate model — one specification, identical for B0 and B1

`score ~ rad_system[=1] + rad_system[=2] + age_group + sex + symptom_count + past_tb`,
ordinary least squares, fitted among TB-negatives. `symptom_count` is the sum of cough,
fever, weight loss and night sweats. Incremental R-squared is the increase when one
concept similarity is added. **In-sample and not cross-validated**, so a small positive
value is expected even for noise.

This is an **association model**. It does not establish that the classifier relies on any
visual cue, and `past_tb` is self-reported clinical history, not a radiographic
annotation.

## 5. Endpoints

**Primary.** Paired change in Spearman rho between classifier score and **scarring**
similarity, among TB-negative cases: **rho(B1) - rho(B0)**.

**Secondary, in this order:**

1. the same paired change for **fibrosis**;
2. paired change in incremental R-squared for scarring and for fibrosis;
3. the same paired changes for the **negative controls**;
4. the same for the remaining 7 primary concepts;
5. stratified by `past_tb` among TB-negatives, with subgroup counts;
6. among TB-**positive** cases: score/concept association, descriptive;
7. **safety:** sensitivity among the 139 TUNE positives at each arm's own **locked
   TBX11K-development threshold** — B0 `-0.5973`, B1 `-2.1911`, recorded in
   `active_tb_label_raydino_20261024_1254/tables/metrics_by_arm.csv` as the smallest score
   reaching >= 0.90 specificity on TBX11K development. **No threshold is fitted on TUNE**;
8. descriptive TUNE AUROC / AUPRC with positive prevalence, both arms;
9. score distributions and missingness.

## 6. Uncertainty

**Paired, case-level bootstrap.** Resampling unit: the **patient** (Vietnam is one image
per patient, so case-level and patient-level coincide). **10,000 replicates, seed
20261031.** The *same* resample indices are applied to B0 and B1 in every replicate, so the
pairing is preserved and the difference is not inflated by independent resampling.

Degenerate-resample handling: a replicate is discarded if the resampled concept vector or
score vector is constant, or if a required subgroup has fewer than 20 members. The number
discarded is reported.

Intervals are the 2.5th and 97.5th percentiles of the replicate differences.

**Multiplicity.** Benjamini-Hochberg across the **9 primary concepts**, on the
TB-negative group only, exactly as in the previous audit. Subgroup and positive-case
p-values are **unadjusted and exploratory** and are labelled so. Negative controls are
outside the family.

## 7. Reproduction gate — the run stops here if it fails

Before any B0/B1 comparison is interpreted, the `baseline_production` column must
reproduce the published 2026-10-23 Kaggle figures, among the 1,213 TB-negatives:

| quantity | published | tolerance |
|---|---|---|
| scarring rho | **0.5661** | +- 0.0005 |
| scarring 95% CI | [0.5276, 0.6020] | +- 0.005 |
| fibrosis rho | **0.4166** | +- 0.0005 |
| scarring incremental R-squared | **0.1975** | +- 0.0010 |
| baseline clinical-only R-squared | **0.2441** | +- 0.0010 |

This is the whole reason the production column is carried. It verifies the checkpoint, the
revision, the preprocessing, the projection, the concept indices and the analysis code in
one number, against a run performed on different hardware. **If it does not reproduce, the
run stops and reports a blocker rather than publishing new associations.**

## 8. Predeclared interpretation rules

**Mechanistic support (exploratory)** requires *all* of:

1. scarring rho is **lower** under B1 than B0 among the same TB-negative cases;
2. the paired 95% CI for that change **excludes zero**, in the expected direction;
3. the incremental-R-squared change is **directionally consistent**;
4. the **negative control does not show a comparable shift** — specifically, the control's
   paired change is smaller in magnitude than the scarring change and its interval does not
   exclude zero in the same direction with comparable size;
5. **no concerning degradation among TB-positive cases** — locked-threshold sensitivity
   does not fall materially, and the positive-case association does not move in a way
   consistent with suppressing true TB.

**No clear mechanism evidence:** intervals overlap zero, findings conflict across
concepts, or the control moves similarly — which would indicate a generalised score
re-ranking rather than a specific one.

**Potential harm:** B1 weakens the association *and* shifts positives in a way consistent
with suppressing true TB, or materially worsens locked-threshold sensitivity, or the
apparent gain traces to another score-distribution change.

Even if every criterion is met, the conclusion is phrased as **support for a mechanism
hypothesis on exposed TUNE data** — not causation, not validation, not clinical benefit,
not novelty.

**The +0.02 AUROC adoption threshold does not apply here.** This is a mechanism analysis,
not a performance-selection exercise. **B1 is not adopted and the production model is not
modified by this run, under any outcome.**

## 9. Hard constraints

- **Vietnam REPORT**: not attached, read, scored, exported or referenced. Enforced by an
  assertion and a test.
- **TBX11K `internal_test`**: not touched. Same enforcement.
- **No per-case export.** Concept similarities are computed into a scratch directory
  outside this run directory and deleted when the run finishes. No per-case concept score,
  image, embedding, clinical row or pid leaves the machine or enters this run directory.
  Only aggregate statistics, counts, intervals and non-identifying run metadata are
  written.
- **Inference runs locally**, not on Kaggle — see `RUN_REPORT.md` §2. This is a
  *tightening* of the data-handling constraint, not a relaxation: no image, clinical row
  or pid is uploaded anywhere at all.
- **No training, fine-tuning, recalibration, threshold selection on TUNE, score
  subtraction, apical weighting or model adoption.**
- **No modification** of production source, tracked docs, existing experiment artifacts,
  checkpoints, prediction files or feature caches. Work only in this run directory and a
  scratch directory.
- **No claim** that `past_tb` equals radiographic fibrosis, that a concept similarity is a
  radiologist diagnosis, or that an association proves the classifier uses a visual cue.
- If a prerequisite fails, the run **stops with a blocker report**. No guard is weakened
  to force completion.
