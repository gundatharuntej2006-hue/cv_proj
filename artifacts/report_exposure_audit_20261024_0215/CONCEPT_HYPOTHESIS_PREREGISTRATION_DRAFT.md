# DRAFT preregistration — concept-informed intervention

> **STATUS: DRAFT. Not an approved protocol. Not active. Nothing in it has been run.**
> It is written so that a decision can be taken against a concrete design rather than
> against an intention.

> **Bottom line up front: the recommendation is NOT to open Vietnam REPORT yet.** §7 gives
> the reason, which is not caution in the abstract — it is that the one endpoint capable
> of detecting harm is too small to detect it. §8 gives what to do instead.

---

## 1. Hypothesis, in neutral terms

> Among participants screened for TB, does a pre-specified, TUNE-derived,
> concept-informed adjustment to the frozen classifier's score change (a) overall TB
> discrimination and (b) the false-positive pattern among Xpert-negative participants with
> previous TB — **in either direction**?

Stated so that "no change" and "worse" are outcomes, not failures. The TUNE evidence
(ρ = 0.566 for scarring, ΔR² = 0.198, scarring-vs-TB AUROC 0.494) motivates the question.
It does not predict the answer.

## 2. Candidate interventions, compared

All four were considered. None is assumed safe or effective.

### A. Fixed subtraction — `score' = score − β · scarring_sim`, β from TUNE negatives

| | |
|---|---|
| appeal | directly removes the axis measured to be large and label-uninformative |
| **fatal objection** | β is estimated on **negatives only** and then applied to everyone. Among positives the same correlation is ρ = 0.238, not 0.566 — the structure differs by class. A coefficient fitted where the effect is strong, applied where it is weak, systematically over-subtracts from positives |
| **clinical risk** | **recurrent TB.** Roughly 36% of Xpert-positives have previous TB. Those are precisely the patients whose radiographs carry both scarring *and* active disease. A scarring-proportional subtraction pushes them down |
| verdict | **rejected.** Also forbidden by the project's standing rule against subtracting concept scores from TB scores |

### B. Embedding orthogonalisation — project the scarring direction out of the 768-d features, refit the head

| | |
|---|---|
| appeal | more principled than score arithmetic; acts before the decision function |
| objection | requires **refitting the classifier**, so it is a new model, not an adjustment. The head is fitted on TBX11K, so the refit happens on a different cohort from where the direction was measured |
| precedent | the closest prior attempt, INLP removal of the **scanner** subspace, was preregistered and **rejected on REPORT** (Δ −0.0002). Representation-level removal has already failed once in this project |
| verdict | **rejected for now** — scope far exceeds an adjustment, and it would invalidate every frozen-model claim in the current write-up |

### C. Conditional recalibration — `logit(TB) ~ score + scarring_sim`, fitted on TUNE

| | |
|---|---|
| form | one logistic regression on TUNE (all 1,352, both classes), two coefficients. The new score is the fitted probability |
| why it is the least unsafe | the **data choose the sign and magnitude**, including zero. If scarring carries no information conditional on the score, the coefficient shrinks toward zero and the ranking is essentially unchanged. The null outcome is "nothing happens", not "harm" |
| it is not subtraction | it is a two-variable monotone re-scoring fitted on both classes, so the class-structure objection to A does not apply |
| residual risk | still **lower** for recurrent TB if the fitted coefficient is negative and those patients sit high on the scarring axis. Mitigated, not eliminated, by fitting on both classes |
| verdict | **the only candidate worth specifying** |

### D. No intervention test

Always on the table. See §7 — it is what this draft recommends.

**Recommended intervention, if one is run: C.** Frozen in full before any REPORT access:
coefficients, the exact TUNE fitting set, the scarring index, the preprocessing, and the
code hash.

## 3. Baseline, cohort, data

| | |
|---|---|
| baseline | the frozen regional-pooling configuration, grid 12 / max / seeds 17–79, as evaluated in `artifacts/p2/regional/mil_final.json`. REPORT AUROC **0.7242** [0.6843, 0.7618] |
| comparator arm | intervention C applied to the same cases, same scores, same order |
| eligible cohort | **Vietnam REPORT, all 1,356 participants, no exclusions.** 141 Xpert-positive |
| concept input | CLEAR scarring similarity, pinned revision `142f169c8b00…`, package commit `57d2eae57dda…`, index from `configs/concept_indices.json`, CXR preprocessing from `clear.build_cxr_preprocess` |
| what must be frozen first | the fitted coefficients, a SHA-256 of the scoring code, and this document, all committed before a single REPORT image is read |
| reserved, untouched | **TBX11K `internal_test` — not read, not scored, not inspected, under any arm.** See the open flag in `UNRESOLVED_ITEMS.md` |

## 4. Endpoints

**Primary.** Change in AUROC on the full REPORT cohort, baseline → C, **paired**.

**Secondary, pre-specified, in this order:**

1. AUROC among Xpert-negatives with previous TB plus all positives (the false-positive
   subgroup framing already used in `prevtb_subgroup_20261021_1055`).
2. Specificity at the operating threshold, overall and in the previous-TB negatives.
3. **Sensitivity among Xpert-positives with previous TB** — the safety endpoint.
4. Referral-rate spread across the three scanners.

**Uncertainty.** Paired bootstrap, 10,000 case-level resamples, seed declared in advance.
Vietnam is one image per patient, so case-level and patient-level resampling coincide.

**Multiplicity.** One primary endpoint, no correction. The four secondary endpoints carry
Benjamini–Hochberg across themselves and are explicitly **not** promotable to primary —
the precedent is `realworld_generalization_20261021_1015`, where a secondary arm improved
on all three endpoints and was correctly **not promoted**.

**Minimum meaningful effect: +0.02 AUROC.** Justification: the paired standard error in
this cohort is ≈ 0.004, so a 95% interval half-width is ≈ 0.008 and +0.02 is resolvable.
The previously declared +0.05 was set for a different question and would make any
realistic result a failure by construction. +0.02 is also just below the exploratory
offset effect (+0.0206), so the bar is not set to flatter the hypothesis.

## 5. Stop / no-go rule, declared before opening REPORT

**Adopt only if all four hold:**

1. primary Δ AUROC **≥ +0.02**;
2. paired 95% CI **excludes zero**;
3. sensitivity among positives with previous TB **does not fall** — point estimate ≥
   baseline, and the lower bound of the paired CI on the difference ≥ −0.02;
4. no scanner's sensitivity falls by more than 0.02.

**Any single failure ends it.** The result is reported as obtained, with no re-analysis,
no endpoint substitution and no second arm. **No acceptable AUROC gain may offset a
sensitivity loss in recurrent TB** — criterion 3 is not tradeable against criterion 1.

## 6. REPORT exposure declaration — to be reproduced verbatim in any write-up

> Vietnam REPORT is **not** an untouched test set. It has been scored in at least six
> analyses across five runs (`report_exposure_ledger.csv`, events E1–E7); at least four of
> those influenced a decision. The previous-TB false-positive pattern that motivates this
> hypothesis was itself characterised on the **pooled 2,708 participants, REPORT
> included** (E6). The nearest prior intervention already has a REPORT point estimate of
> +0.0206 (E3), which is currently acting as a design parameter elsewhere in the project.
>
> Any result from this protocol is therefore a **one-time, hypothesis-specific
> exploratory test on an already-exposed cohort**. It is **not** independent external
> validation and must never be described as such.

## 7. Why the recommendation is to NOT run this yet

**The safety endpoint cannot do its job.** REPORT contains roughly **51** Xpert-positives
with previous TB (141 positives × 36.4%). At that size:

- a sensitivity estimate carries a 95% half-width of about **±0.13**;
- a clinically important 5-percentage-point loss is **about 3 patients**;
- under a paired McNemar comparison, 3 discordant cases in one direction is not
  distinguishable from chance.

So criterion 3 would be satisfiable by a design that is blind to the exact harm it
exists to prevent. Passing it would mean "we could not see a loss", not "there was none".
Writing that into a protocol as a safety guarantee would be misleading.

**And the upside is smaller than it looks.** Because REPORT is already exposed (§6), a
result there is not validation. Most of what this test would buy — an honest effect
estimate under a frozen design — is obtainable from TUNE by nested cross-validation,
without spending anything.

**Two scientific questions are also still open**, and both bear directly on whether C is
even sensible:

- **The consolidation anomaly.** ρ = −0.300 among negatives and **−0.385 among positives**
  — the score falls as consolidation similarity rises, in the group that has TB.
  Consolidation is the canonical active-TB finding. Until that is understood, the nine
  axes cannot be read as radiological findings, and an intervention built on one of them
  rests on an interpretation that the data partly contradict.
- **Cavity was never measured** — no exact vocabulary entry — and it is the finding that
  most distinguishes active from healed disease. The concept set is missing the variable
  most relevant to the safety concern.

## 8. What to do instead, in order

1. **Nested cross-validation of intervention C within TUNE.** Fit and evaluate in
   held-out folds. Costs nothing, touches no reserved data, and gives an honest estimate
   of the effect and its sign.
2. **Re-run the concept audit with a negative-control concept** and an exported **9×9
   concept correlation matrix** (aggregate, not per-case). This tests whether the nine
   axes are near-collinear and whether an unrelated concept also correlates with the
   score. TUNE only; the existing package runs it unchanged.
3. **Resolve the consolidation anomaly**, or document that it cannot be resolved and treat
   the concept names as uninterpretable labels.
4. **Settle the `internal_test` flag** (`UNRESOLVED_ITEMS.md`) with P1/P4.
5. **Only then** decide whether to spend REPORT. If steps 1–3 show a stable effect of at
   least +0.02 with no sign of recurrent-TB harm, this draft can be finalised, frozen and
   executed as written. If they do not, no REPORT read is warranted and the honest output
   is the diagnostic finding on its own, which is publishable as a negative/descriptive
   result.

## 9. Scope limits that hold regardless

- No concept score may be subtracted from, or used to suppress, any referral.
- No threshold may be fitted on REPORT.
- TBX11K `internal_test` is not read under any arm.
- `past_tb` is clinical history, not an image annotation.
- A concept score is a weakly supervised text-similarity proxy, not a detector.
- Nothing here is a clinical-readiness, novelty or causality claim.
