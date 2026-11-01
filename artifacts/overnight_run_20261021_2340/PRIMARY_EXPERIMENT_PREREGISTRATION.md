# Primary experiment: source-balanced head vs. standard head

**Status: BLOCKED. No model was fitted. No data was re-partitioned.**

This document was written before any new result was looked at, and it records why the
pre-registered experiment cannot be run under the project's own split protocol rather than
improvising a version that could be.

---

## 1. The question the experiment was to answer

> Can a simple source-balanced training strategy for the existing lightweight head improve
> source-robust discrimination, without changing the frozen RAD-DINO representation or the
> regional max-pooling method?

Both arms would share the frozen backbone, the 12×12 regional representation, the optimiser,
the initialisation and seed policy, the regularisation and the training budget. The only
difference would be that Arm B weights or resamples training rows by **source × class**.

## 2. Why it is blocked

**The training data has no source variable, and the data that has one may not be trained on.**

### 2.1 The permitted training pool carries no acquisition source

`artifacts/p2/labels_tbx11k_trainval_v1.csv` is the label file behind every head fitted in
this project. Its complete column list is:

```
image_id, rel_path, official_split, tbx11k_folder, tbx11k_category,
tb_label, active_tb_label, has_boxes, n_boxes, box_cats
```

There is **no scanner, device, manufacturer, institution, site or acquisition-date field**,
and none is distributed with TBX11K. `tbx11k_folder` and `tbx11k_category` encode *class*
(`health` 3,800 · `sick_non_tb` 3,800 · `tb` 800), not source. Using them as the grouping
variable would be class balancing relabelled as source balancing, which would misdescribe
the experiment.

A source-weighted objective needs group counts. Here the number of groups is one. **Arm B
reduces exactly to Arm A**, and the comparison is empty.

### 2.2 The only cohort with a source variable is an evaluation cohort

The Vietnam Dryad cohort is the sole dataset in the project with a genuine acquisition-source
column, `rad_system ∈ {1, 2, 3}`:

| rad_system | n | TB | prevalence | in TUNE | in REPORT |
|---|---|---|---|---|---|
| 1 | 408 | 44 | 0.108 | 204 | 204 |
| 2 | 1,332 | 151 | 0.113 | 665 | 667 |
| 3 | 968 | 85 | 0.088 | 483 | 485 |
| **all** | **2,708** | **280** | **0.103** | 1,352 | 1,356 |

Its two partitions have declared roles: **TUNE** is development and chose the pooling
configuration; **REPORT** is reserved and was read once. Fitting a tuberculosis head on
either would:

1. convert the project's only consecutive screening cohort with an Xpert reference standard
   from an evaluation set into training data, which destroys the transfer-gap result the
   whole report rests on; and
2. require redefining a declared split role, which the runbook forbids outright.

Leave-one-scanner-out over `rad_system` is therefore unavailable: every fold's training rows
would have to come from Vietnam.

### 2.3 The stated fallback is unavailable for the same reason

The fallback is *"source-aware patient-grouped cross-validation within the permitted
training/development data."* The permitted training/development data is TBX11K train and
TBX11K development. Neither has a source variable (§2.1), so source-aware CV cannot be
constructed there either.

### 2.4 The remaining external cohorts cannot supply the contrast

Shenzhen (662), Montgomery (138) and DA/DB (176) are each effectively a single site, so
"source" is perfectly confounded with "dataset", and all three are barred from fitting —
Shenzhen and Montgomery because they already informed preprocessing and negative-composition
choices, DA/DB because it is an untouched external site worth keeping that way.

### 2.5 Verdict

Per the runbook's own stop rule — *"or skip fitting and report why"* — **fitting was skipped.**
No surrogate source variable was invented, because inventing one is forbidden and would make
any resulting number uninterpretable.

---

## 3. What was measured instead, with no fitting

Two read-only analyses bear directly on the primary question and were run in its place.

### 3.1 Source is almost perfectly decodable from the frozen representation

A diagnostic probe (multinomial logistic regression on standardised 768-d globally pooled
frozen embeddings, 5-fold CV, Vietnam **TUNE only**, REPORT never opened):

| quantity | value |
|---|---|
| balanced accuracy predicting `rad_system` | **0.9737** |
| macro F1 | 0.9772 |
| majority-class accuracy | 0.4919 |
| balanced accuracy with scanner labels permuted (10 repeats) | 0.3381 mean, 0.3515 max |

Confusion matrix (rows true 1/2/3, columns predicted):

```
[[195,   6,   3],
 [  2, 657,   6],
 [  1,  10, 472]]
```

**Interpretation, stated carefully.** The frozen features retain nearly all the information
needed to identify which of three radiography systems produced an image. That is what a
source-balancing method would have to work against, and it establishes that the signal such
a method targets is present and strong.

**It does not show that the tuberculosis classifier uses that information.** A representation
can encode a nuisance variable that the downstream head ignores. The probe is evidence of
*availability*, not of *use*. Separately, scanner and TB status are only weakly associated in
this cohort (χ² = 4.051, df = 2, p = 0.132, Cramér's V = 0.039), so the scanner signal is not
a proxy for the outcome here.

### 3.2 The baseline any source-balanced head would have to beat is now measured

Scanner-stratified performance of the **existing** focal-pooling head (12×12 max, seeds
17/29/43/61/79, trained on TBX11K only), recomputed from the committed case-level score file
with no refitting:

| scanner | n | TB | prevalence | AUROC | Lunit on the same patients |
|---|---|---|---|---|---|
| rad_system=1 | 408 | 44 | 0.108 | 0.7114 | 0.7809 |
| rad_system=2 | 1,332 | 151 | 0.113 | 0.7820 | 0.8095 |
| rad_system=3 | 968 | 85 | 0.088 | 0.7000 | 0.7422 |
| all | 2,708 | 280 | 0.103 | 0.7240 | 0.7827 |

Worst-source AUROC is **0.7000** on rad_system=3. Any future source-robustness claim must
improve that number, not the pooled one. Per-scanner intervals are in
`baseline_reproduction.csv`; with three scanners, none of this supports a population-level
claim about scanners in general.

---

## 4. The design that would be permissible, recorded now so it cannot be retrofitted

If this is picked up later, the following is pre-registered. It is written before any
corresponding result exists.

**Fit only a source-removal projection, never a tuberculosis head, on Vietnam.**

- Keep the TB head trained on **TBX11K train** exactly as now (healthy-only negatives,
  lungs-only masking, 12×12 max, seeds 17/29/43/61/79). It is not retrained.
- On **Vietnam TUNE only**, fit a linear subspace that predicts `rad_system` from the 768-d
  token representation, and project that subspace out of the tokens before pooling.
  Scanner labels are not the outcome variable, and TUNE is the declared development half, so
  no outcome information leaks.
- Evaluate on **Vietnam REPORT**, once, reporting per-scanner and pooled AUROC with paired
  bootstrap against the unprojected head on identical cases.
- Decision rule, fixed in advance: accept only if the **worst-scanner** AUROC improves and
  the paired pooled difference excludes zero. A pooled gain with worst-scanner degradation
  is a rejection, not a win.
- Predeclared seeds: 17, 29, 43, 61, 79. Subspace rank fixed at 3 (the number of scanner
  classes minus one, plus one) before any result is seen.

**Why it was not run tonight:** it needs `regional_tbx11k_lungs_g12_v1.npz` (1.35 GB) and
`regional_vietnam_lungs_g12_v1.npz` (555 MB) resident. Available RAM throughout this run was
**0.56–0.8 GB of 16.4 GB**. The runbook caps work at the available compute, so it was not
attempted. It is streamable in chunks and is the recommended next step.

**It also remains exploratory**, because the Vietnam cohort already influenced the pooling
choice. It cannot produce an independent generalisation estimate, only a cleaner comparison.
