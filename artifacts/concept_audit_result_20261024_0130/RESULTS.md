# CLEAR concept audit — results

**Run:** Kaggle, Tesla T4, 2026-10-23T20:15:28 UTC, 9.3 min wall clock.
**Cohort:** Vietnam **TUNE only**, 1,352 cases (139 Xpert-positive, 466 previous TB),
fingerprint `1aa55cd5c0eb2088…` matching the pinned value.
**Model:** CLEAR, pinned HF revision `142f169c8b00…`, loaded by `clear.load_pretrained`
at commit `57d2eae57dda…`; all three asset checksums matched.

> **The screening AUROC is unchanged at 0.7242.** This run measured nothing about
> screening performance and did not touch the REPORT partition.

---

## 1. The finding

Among the **1,213 TB-negative** participants, the frozen classifier's score is strongly
rank-correlated with the *scarring* and *fibrosis* concept directions. 8 of 9 concepts
survive Benjamini–Hochberg at 0.05; only pleural thickening does not.

| concept | ρ (negatives, all) | 95% CI | p_BH | ΔR² over scanner+age+sex+symptoms+past_tb |
|---|---|---|---|---|
| **scarring** | **+0.566** | 0.528 – 0.602 | 9.0e-103 | **+0.198** |
| **fibrosis** | **+0.417** | 0.369 – 0.463 | 1.8e-51 | +0.100 |
| **consolidation** | **−0.300** | −0.351 – −0.248 | 3.5e-26 | +0.018 |
| calcified granuloma | −0.165 | −0.218 – −0.110 | 1.3e-08 | **+0.000** |
| volume loss | +0.170 | 0.115 – 0.223 | 5.8e-09 | +0.038 |
| bronchiectasis | +0.082 | 0.027 – 0.137 | 6.3e-03 | +0.010 |
| calcification | +0.080 | 0.025 – 0.135 | 6.3e-03 | +0.021 |
| focal opacity | −0.080 | −0.134 – −0.023 | 6.3e-03 | +0.008 |
| pleural thickening | +0.051 | −0.003 – 0.106 | 0.075 **(ns)** | +0.007 |

Baseline R² is **0.2441**; adding scarring alone takes it to **0.4416**.

## 2. Why this is unlikely to be a global artifact

**The pattern is differentiated.** A single nuisance factor — exposure, contrast, scanner,
body habitus — would move all nine concepts together. It does not:

- consolidation is **negative**, and strongly so;
- pleural thickening is **null**;
- calcified granuloma is associated marginally (ρ = −0.165) but adds **ΔR² = 0.0000** once
  covariates are in. Its association is fully explained by scanner and history. Scarring's
  is not.

That internal contrast is doing the work a negative-control concept should have done.
**No orthogonal control concept was included** — all nine are TB-adjacent — and that is a
design omission, not a strength of the result.

## 3. The stratification changes the earlier interpretation

My first reading was that this is the known previous-TB effect showing a visual
correlate. The stratified table says otherwise:

| concept | neg, all | neg, **past TB** (417) | neg, **no past TB** (796) | positives (139) |
|---|---|---|---|---|
| fibrosis | 0.417 | 0.372 | 0.397 | 0.218 |
| **scarring** | 0.566 | 0.418 | **0.570** | 0.238 |
| volume loss | 0.170 | 0.255 | 0.082 | 0.192 |
| calcified granuloma | −0.165 | −0.257 | −0.107 | −0.218 |
| consolidation | −0.300 | −0.285 | −0.288 | −0.385 |
| bronchiectasis | 0.082 | −0.092 | 0.148 | −0.122 |

The scarring association is **as strong, in fact stronger, in negatives with no previous
TB** (0.570, n=796) than in those with it (0.418, n=417). So this is **not** the past-TB
subgroup effect wearing a new hat. The score tracks a chronic-appearance axis across
TB-negative participants generally, regardless of recorded history — consistent with
scarring adding ΔR² = 0.198 on top of a baseline that already contains `past_tb`.

Separately, the distributions do show past-TB negatives looking more "scarred"
(mean 0.408 vs 0.377). Both things are true; they are not the same claim.

## 4. The number that matters most for what comes next

From `concept_discrimination.csv`, each concept score as a predictor of Xpert status:

| concept | AUROC vs TB label | 95% CI |
|---|---|---|
| bronchiectasis | 0.620 | 0.570 – 0.669 |
| volume loss | 0.599 | 0.554 – 0.648 |
| fibrosis | 0.579 | 0.528 – 0.636 |
| focal opacity | 0.515 | 0.468 – 0.561 |
| **scarring** | **0.494** | **0.445 – 0.542** |
| consolidation | 0.469 | 0.423 – 0.515 |
| calcification | 0.419 | 0.372 – 0.467 |
| calcified granuloma | 0.392 | 0.343 – 0.450 |
| pleural thickening | 0.336 | 0.288 – 0.383 |

**Scarring is the strongest predictor of the classifier's score and is pure chance as a
predictor of TB.** ρ = 0.566 with the score, AUROC 0.494 with the label, CI straddling
0.5. That is the signature of a **nuisance direction**: a large, systematic component of
the score that carries no information about the outcome.

## 5. Can this raise the screening AUROC?

**Not from this run, and not yet.** Three separate statements, which should not be
collapsed:

**(a) This run cannot.** It is descriptive. Nothing was fitted, nothing was changed, and
the 0.7242 figure was not recomputed. Any claim otherwise would be false.

**(b) There is now a principled reason to think an improvement is available.** If the
score decomposes roughly as *TB signal + scarring direction + noise*, and the scarring
direction is uninformative about the label (§4), then removing its contribution removes
variance that is pure noise with respect to Xpert status. That should raise discrimination
rather than lower it. This is the first mechanism this project has identified that is both
**large** (ΔR² = 0.198) and **demonstrably label-uninformative**.

**(c) It is forbidden under the current rules, and the rule should be revisited
deliberately rather than quietly.** The standing constraint is explicit: *do not use a
concept score to subtract from active-TB scores or suppress referrals.* That rule was
written before this evidence existed. The honest move is to put the new evidence against
it and decide, not to route around it.

### Reasons for caution that survive the argument above

1. **Marginally uninformative is not conditionally uninformative.** AUROC 0.494 says
   scarring alone does not predict TB. It does not prove scarring carries no information
   *given* the score. Residualising could remove real signal.
2. **The class-conditional strengths differ**: ρ = 0.566 in negatives, 0.238 in positives.
   A single global adjustment does not fit that structure, and naive subtraction could
   depress true positives.
3. **Consolidation is the warning sign.** ρ = −0.300 among negatives and **−0.385 among
   positives** — the score goes *down* as consolidation-similarity goes up, in the group
   that actually has TB. Consolidation is the canonical active-TB finding. Either these
   concept directions do not mean what their names say, or something is inverted. Until
   that is understood, building an intervention on these axes is premature.
4. **TUNE is exposed.** Anything fitted here and evaluated here is circular.
5. **Cavity was never measured**, and it is the finding that most distinguishes active
   from healed disease.
6. **CLEAR's exposure to this cohort is unknown.**

### Honest expectation of size

Unknown, and I will not invent a figure. The project is precision-limited rather than
idea-limited (paired SE ≈ 0.004 against single-arm SE ≈ 0.019), so a paired design can
resolve changes of about 0.01 AUROC. Whether removing a nuisance axis worth ΔR² = 0.198 of
*score variance* converts into 0.00 or 0.03 of *AUROC* cannot be predicted from these
numbers — score variance and ranking performance are not the same quantity.

## 6. Against the project's decision rules

`DECISION_RULES.md` in `kaggle_concept_raydino_20261021_1830` requires all four conditions
for a signal to be actionable. This result **does not clear that bar**, principally on
condition 4 (independent confirmation): everything here is TUNE, and TUNE is exposed.

What has changed is that the hypothesis is now **specific enough to preregister**, which
it was not before.

## 7. What was and was not done

| | |
|---|---|
| concept scores computed | **yes** — (1352, 9), all finite |
| partition used | Vietnam **TUNE only** |
| Vietnam REPORT | **not read, not scored, not attached** |
| TBX11K `internal_test` | **not referenced** |
| any model, checkpoint or prediction changed | **no** |
| any screening metric recomputed | **no** — 0.7242 stands |
| per-case concept scores exported | **no** — shards stayed in the Kaggle session |
| multiplicity | BH across 9 concepts, `negatives (all)` family only; subgroup p values are uncorrected and exploratory |

Environment drift from the pins is recorded in `outputs/tables/preflight_kaggle.json`:
numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.16.3, torch 2.11.0 against pins of
2.2.6 / 3.0.3 / 1.9.0 / 1.17.1 / 2.6.0. Rank correlations and bootstrap intervals are not
sensitive at this level, but it is recorded rather than waved away.

## 8. Standing interpretation limits

A similarity to the sentence "fibrosis" is **not** fibrosis detection. Each score is a
weakly supervised text-similarity proxy, not a validated detector and not radiologist
ground truth. `past_tb` is clinical history, not an image annotation. Association is not
causation: none of this shows the classifier *uses* a particular visual cue. Nothing here
is a clinical-readiness, novelty or causality claim.
