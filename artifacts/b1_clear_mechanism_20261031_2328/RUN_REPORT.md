# B1 label alignment x CLEAR mechanism test — run report

**Run:** `artifacts/b1_clear_mechanism_20261031_2328/`
**Executed:** 2026-10-31, 23:31 -> 2026-11-01 00:09 +05:30 (inference + analysis 35.2 min)
**Git HEAD before and after:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` — **unchanged**
**Analysis manifest sha-256:** `eea6bfcad3d1a025cec3f04d6b86d59a2c1c8aa4d2e871c2acbf2a5015e509f9`,
hashed 2026-10-31T18:05:29Z — **before** any inference was run or any association seen

**Classification: exploratory mechanism analysis on an exposed cohort.** Not independent
validation. **B1 was not adopted and no production artifact was modified.**

---

## Verdict: **NO CLEAR MECHANISM EVIDENCE**

Three of the five predeclared criteria are met, cleanly and with tight intervals. The
fourth is met on the primary negative control and **fails on the secondary**. The deciding
observation is broader than either control: **8 of the 11 encoded concepts shifted with
paired intervals excluding zero**, 7 of them among the 9 primary ones. That is the signature of a generalised re-ranking of
the score, not of one axis being specifically removed.

| # | predeclared criterion | result |
|---|---|---|
| 1 | scarring rho **lower** under B1 | **met** — 0.5661 -> 0.5317 |
| 2 | paired 95% CI **excludes zero**, expected direction | **met** — [-0.0428, -0.0265] |
| 3 | incremental R-squared **directionally consistent** | **met** — -0.0250 [-0.0315, -0.0184] |
| 4 | negative control shows **no comparable shift** | **SPLIT** — `scoliosis` passes, `degenerative changes` fails |
| 5 | **no concerning degradation** among TB-positives | **marginal** — sensitivity 0.9424 -> 0.9281 at the locked threshold, 131 -> 129 of 139 |

The predeclared rule for mechanistic support is conjunctive. It is not satisfied.

---

## 1. Did the analysis execute?

**Yes, in full, and it passed its own reproduction gate before any new association was
examined.**

| | |
|---|---|
| cohort | Vietnam TUNE, **1,352** patients, **139** Xpert-positive, **466** with previous TB |
| cohort fingerprint | `1aa55cd5c0eb2088f2a037d9a8e5770ca1c0f4a75ea21566d743d17577ed1751` — matches the pinned staging manifest |
| images | 1,352 approved, **0 missing**, 1,356 present on disk but **not approved and never opened** |
| encoder | CLEAR, HF revision `142f169c8b00…`, package commit `57d2eae57dda…`, `clear.load_pretrained`, visual tower **87,174,144 parameters** |
| preprocessing | CLEAR's own `build_cxr_preprocess` — bicubic 448 px, CXR statistics. Not reimplemented |
| assets | `best_model.pt` **sha-256 `da1617a7…`**, `concept_embeddings_368294.pt` **`7241c5b9…`**, `mimic_concepts.csv` **`93e975b7…`** — all three hashed against the pinned record and matching, byte counts exact |
| hardware | local RTX 4050 Laptop GPU, torch 2.6.0+cu124, Python 3.13.2, Windows 11 |
| encoding | 1.94 min; analysis and bootstraps 33.3 min |

### 1a. Deviation from the brief, declared: inference ran locally, not on Kaggle

The brief assumed a private Kaggle session. **This ran entirely on the local machine.**
That is a *tightening* of the data-handling constraint, not a relaxation: no image, no
clinical row and no patient identifier was uploaded anywhere at all, and no Kaggle dataset
was created. The only network traffic was the download of the pinned CLEAR weights from
Hugging Face.

It also bought something the Kaggle route could not: because the run happens on different
hardware from the 2026-10-23 audit, reproducing that audit's numbers becomes a real check
rather than a tautology. See §2.

## 2. Reproduction gate — passed to five decimal places

Before any B0/B1 comparison was computed, the production score column was pushed through
the whole pipeline and checked against the published Kaggle figures:

| quantity | Kaggle 2026-10-23 (T4) | local (RTX 4050) | difference |
|---|---|---|---|
| scarring rho, TB-negatives | 0.5661 | **0.5661** | 0.000008 |
| scarring 95% CI low | 0.5276 | 0.5276 | 0.000023 |
| scarring 95% CI high | 0.6020 | 0.6019 | 0.000050 |
| fibrosis rho | 0.4166 | **0.4166** | 0.000035 |
| scarring incremental R-squared | 0.1975 | **0.1975** | 0.000000 |
| clinical-only R-squared | 0.2441 | **0.2441** | 0.000000 |

Different GPU, different OS, different torch build, five-decimal agreement. The
checkpoint, revision, preprocessing, projection, concept indices and analysis code are
confirmed identical to the run this builds on. `tables/reproduction_gate.csv`.

### 2a. B0 turned out to **be** the production baseline, not merely resemble it

The manifest carried `score_baseline` separately from B0 (arm A) on the assumption that a
refit would be close but not identical. It is identical: **Spearman 1.000000, Pearson
1.000000, maximum absolute difference 1e-06** — the rounding of the stored CSV at six
decimal places. Arm A is a deterministic reproduction of the production fitting path
(same frozen features, same five seeds, same configuration).

Consequence, stated plainly: the three "arms" in the output tables are really two. The
gate and the comparator are the same vector. This **strengthens** the contrast — B1 is
being compared against production itself, not against an approximation of it — but it
means the gate does not independently validate B0 as a separate quantity, and the
`baseline_production` rows in `tables/aggregate_results.csv` are redundant with the `B0`
rows rather than confirming them.

## 3. Did the scarring and fibrosis associations change?

**Yes, both fell, and both intervals exclude zero.** Among the same 1,213 TB-negative
cases, paired bootstrap, 10,000 replicates, seed 20261031, identical resample indices
applied to both arms, **0 degenerate resamples discarded**:

| concept | rho B0 | rho B1 | **delta** | 95% CI | excludes 0 |
|---|---|---|---|---|---|
| **scarring** | 0.5661 | 0.5317 | **-0.0344** | **[-0.0428, -0.0265]** | yes |
| **fibrosis** | 0.4166 | 0.3984 | **-0.0182** | **[-0.0269, -0.0103]** | yes |

Incremental R-squared over scanner, age group, sex, symptom count and `past_tb`:

| concept | B0 | B1 | **delta** | 95% CI | excludes 0 |
|---|---|---|---|---|---|
| **scarring** | 0.1975 | 0.1725 | **-0.0250** | **[-0.0315, -0.0184]** | yes |
| **fibrosis** | 0.0996 | 0.0891 | -0.0105 | [-0.0157, -0.0055] | yes |

Scarring's unique contribution falls by 0.025, a **12.7% relative reduction**, and is by
far the largest incremental-R-squared movement of any concept — more than twice fibrosis,
the next largest.

**And the effect concentrates where the hypothesis says it should.** Among TB-negatives
*with* previous TB the scarring association falls by **-0.0512**; among those *without*,
by **-0.0274** (`tables/subgroup_associations.csv`, unadjusted and exploratory). The
subgroup the intervention targets moves about twice as much as the one it does not.

**But the reduction is modest in absolute terms.** rho 0.566 -> 0.532 leaves the
association overwhelmingly intact. Whatever B1 does, it does not remove the chronic-
appearance axis; it nudges it.

## 4. Negative-control behaviour — the result that decides the verdict

| control | rho B0 | rho B1 | delta | 95% CI | excludes 0 | delta incremental R-squared |
|---|---|---|---|---|---|---|
| **`scoliosis`** (primary) | 0.0184 | 0.0114 | **-0.0070** | **[-0.0153, +0.0013]** | **no** | -0.0001 [-0.0012, +0.0008] |
| `degenerative changes` (secondary) | -0.1381 | -0.1764 | **-0.0384** | [-0.0475, -0.0297] | **yes** | **+0.0070** [+0.0033, +0.0113] |

**The primary control behaves exactly as a control should.** Its interval covers zero, its
shift is a fifth of scarring's, and its incremental R-squared is flat at -0.0001. It was
also chosen *before* anything was measured, and the concept-correlation matrix computed
afterwards shows it is genuinely orthogonal to the axis under test: **`scoliosis` x
`scarring` = -0.005**. That is as clean as a negative control gets.

**The secondary control does not.** Its signed shift (-0.0384) is **larger than scarring's**
(-0.0344), its interval excludes zero, and its incremental R-squared **rises** by 0.0070
with an interval excluding zero — it *gained* unique explanatory power under B1.

A fair reading has to note that the two controls fail in different ways. `degenerative
changes` has a negative rho that became *more* negative, so its association **strengthened**
in magnitude, whereas scarring's moved *toward* zero. Those are not the same phenomenon,
and the manifest predicted this control would be weak because it tracks age, a covariate.
The measured matrix confirms it is not independent: `degenerative changes` x `scarring`
= +0.246, x `calcification` = +0.723.

**Even so, criterion 4 as written is not met**, and the honest reason is not the control
itself — it is §5.

## 5. The deciding observation: eight of eleven concepts moved

| concept | delta rho | 95% CI | excludes 0 |
|---|---|---|---|
| `degenerative changes` **(control)** | -0.0384 | [-0.0475, -0.0297] | yes |
| pleural thickening | -0.0350 | [-0.0440, -0.0263] | yes |
| **scarring** | **-0.0344** | [-0.0428, -0.0265] | yes |
| calcification | -0.0297 | [-0.0389, -0.0210] | yes |
| **fibrosis** | -0.0182 | [-0.0269, -0.0103] | yes |
| bronchiectasis | **+0.0177** | [+0.0092, +0.0263] | yes |
| volume loss | -0.0111 | [-0.0192, -0.0031] | yes |
| focal opacity | -0.0106 | [-0.0193, -0.0021] | yes |
| calcified granuloma | -0.0077 | [-0.0167, +0.0013] | no |
| **`scoliosis` (control)** | **-0.0070** | [-0.0153, +0.0013] | **no** |
| consolidation | +0.0067 | [-0.0016, +0.0149] | no |

**Scarring is not the largest mover.** It is third, behind a negative control and behind
pleural thickening — whose baseline association (0.0512) is near zero, so its shift is
noise being re-ranked rather than structure being removed.

With 1,213 cases and 10,000 shared-index replicates, intervals of width ~0.017 will
exclude zero for almost any systematic change. **The breadth of the movement, not its
size, is what argues against specificity:** B1 re-orders the score against most of the
concept space at once.

The scale-aware metric tells a more favourable story — in incremental R-squared, scarring
(-0.0250) is more than twice any other concept, and the primary control is flat. That is
the strongest single piece of evidence *for* the hypothesis in this run, and it is why the
verdict is "no clear evidence" rather than "refuted".

### 5a. The nine concepts were never nine independent tests

Computed from this run (`tables/concept_correlation_matrix.csv`, aggregate only):

| pair | Spearman |
|---|---|
| focal opacity x consolidation | **+0.797** |
| focal opacity x volume loss | +0.785 |
| calcification x focal opacity | +0.749 |
| `degenerative changes` x calcification | +0.723 |
| volume loss x consolidation | +0.688 |
| calcification x pleural thickening | +0.685 |
| **scarring x fibrosis** | **+0.625** |
| **`scoliosis` x scarring** | **-0.005** |

This settles item **U2** of `report_exposure_audit_20261024_0215/UNRESOLVED_ITEMS.md`: the
concepts are heavily inter-correlated, the effective number of independent tests is far
below nine, and the Benjamini-Hochberg correction applied across them was **conservative**
for a family that is not independent. No conclusion in the earlier audit changes — the
headline p-value was ~9e-103 — but the family must be described accurately in any write-up.

It also explains part of §5 directly: with concepts this correlated, moving one necessarily
moves several.

## 6. Multiplicity

Benjamini-Hochberg was applied across the **9 primary concepts** on the TB-negatives group
only, separately per arm, exactly as predeclared. The two negative controls are **outside**
the family and carry no adjusted p-value. Subgroup and TB-positive p-values are
**unadjusted and exploratory** and are labelled so in `tables/subgroup_associations.csv`.

Under both B0 and B1 the scarring association survives correction by an enormous margin
(p_BH ~ 9e-103 and ~1.6e-93 respectively).

## 7. TB-positive cases — safety

**Discrimination (descriptive; TUNE is exposed):**

| arm | AUROC | AUPRC | prevalence |
|---|---|---|---|
| B0 | 0.7237 | 0.2007 | 0.1028 |
| B1 | **0.7441** | **0.2241** | 0.1028 |

**At each arm's own locked TBX11K-development threshold** (>= 0.90 specificity on TBX11K
development; **no threshold was fitted on TUNE**):

| arm | threshold | sensitivity | specificity | referral rate |
|---|---|---|---|---|
| B0 | -0.5973 | **0.9424** (131/139) | 0.3759 | 0.6568 |
| B1 | -2.1911 | **0.9281** (129/139) | 0.3792 | 0.6524 |

**Sensitivity falls by 0.0143 — two patients of 139.** That is small, it is in the
direction of concern, and it is **underpowered**: with 139 positives a two-case difference
is well inside sampling noise, and no paired test on it would be informative. Specificity
and referral rate are effectively unchanged.

Two caveats on this operating point. First, a TBX11K-fitted threshold transfers badly to
Vietnam — specificity lands at 0.38, not 0.90 — so this is a comparison of two arms at a
transferred threshold, not a clinically meaningful operating point. Second, B1's absolute
scores sit on a different level (mean among negatives without past TB: -0.10 for B0,
-1.81 for B1), which is why the thresholds differ so much; the per-arm threshold absorbs
that shift, as it must for the comparison to be about ranking rather than calibration.

**Among TB-positives**, the scarring association also falls (0.2375 -> 0.1890) and
consolidation moves slightly toward zero (-0.3853 -> -0.3630). Nothing here looks like
systematic suppression of true TB: the positive-case score distribution shifts down by
about the same amount as everyone else's, and the locked threshold is refitted per arm.
**But the direction of the sensitivity change is the one the guardrail exists to catch,
and this cohort cannot resolve it.**

## 8. Guardrail failures

| guard | outcome |
|---|---|
| analysis-manifest hash unchanged since preregistration | **pass** |
| reproduction gate | **pass**, all six quantities |
| asset sha-256 (checkpoint, embeddings, vocabulary) | **pass**, all three |
| concept indices re-derive from the pinned vocabulary | **pass** |
| cohort fingerprint matches the staging manifest | **pass** |
| 1,352 cases, no duplicate, no missing, no non-finite score | **pass** |
| arm label columns agree with the audited cohort | **pass** |
| Vietnam REPORT pids in cohort | **0** |
| TBX11K `internal_test` referenced | **no** |
| threshold fitted on TUNE | **none** |
| per-case export | **none** — shards computed in scratch, deleted at the end |
| degenerate bootstrap resamples | **0 of 10,000** |
| **predeclared criterion 4 (negative control)** | **FAILED on the secondary control** |
| **predeclared criterion 5 (positive-case safety)** | **marginal** — see §7 |

No guard was weakened. No run was repeated to obtain a different number.

## 9. What this does and does not support

**Supports, as an exploratory mechanism observation on exposed data:**

- B1 reduces the scarring association, with a tight interval, and reduces scarring's
  unique explanatory contribution more than any other concept's.
- The reduction concentrates in TB-negatives with previous TB (-0.0512) over those without
  (-0.0274) — the subgroup the hypothesis names.
- The primary negative control, verified orthogonal to scarring (-0.005), does not move.

**Does not support:**

- That the scarring "shortcut" has been removed. 94% of the association survives.
- That the effect is **specific**. Eight of eleven concepts shifted with intervals
  excluding zero, including a negative control and including concepts with no
  relationship to the training-label change.
- That the +0.0204 TUNE AUROC gain is *caused* by reduced scarring reliance. The two
  co-occur; nothing here establishes a causal path, and a generalised re-ranking would
  produce both.
- Any claim about causation, independent validation, clinical benefit or novelty.
- Adoption of B1. **The +0.02 AUROC adoption rule was explicitly not applied here**, and
  this run changes nothing about the production system.

## 10. Limitations

- **Exposed cohort.** Vietnam TUNE selected the pooling configuration and the concept
  hypothesis was generated from the pooled 2,708 participants. Nothing here is independent.
- **Association, not mechanism.** The covariate model is observational and in-sample. A
  change in association does not demonstrate that a visual cue was removed, or that the
  classifier ever relied on one.
- **Concept scores are weakly supervised proxies.** Each is text-similarity to a single
  MIMIC report sentence. Not a radiologist annotation, not a detector.
- **`past_tb` is self-reported clinical history**, not a radiographic finding of fibrosis.
- **Cavity, nodule and reticular opacity remain unmeasured** — no exact vocabulary entry,
  and deliberately not approximated. Cavity is the finding that most distinguishes active
  from healed disease, so the concept set is missing the variable most relevant to the
  safety question.
- **The consolidation anomaly is unresolved** (item U3): rho = -0.300 among negatives and
  -0.385 among positives, under both arms. Until it is understood the nine axes cannot be
  read as radiological findings, and this entire analysis is a measurement of something
  partly unnamed.
- **CLEAR's exposure to this cohort is unknown** (item U4) — the model card names three
  training corpora and no evaluation set. Not resolvable from any artifact here.
- **The safety endpoint is underpowered**, as it was in the earlier preregistration: 139
  positives cannot resolve a two-patient sensitivity change.

## 11. Production state

**Vietnam screening AUROC remains 0.7242.** It was not recomputed, because Vietnam REPORT
was not read.

No checkpoint, prediction file, threshold, feature cache, tracked document or prior
experiment artifact was created, modified or deleted. Git HEAD is `f878a6c`, unchanged;
`git status` is identical before and after except for this new untracked directory. The
audited package `kaggle_exec_prep_20261023_1130` was **imported and not edited** — its
file hashes are unchanged (`tables/protected_hashes_before.txt`, re-verified by test).

The two negative controls were added by shadowing that package's asset lookup with this
run's own `configs/concept_indices.json`, in which **the nine pinned indices are copied
verbatim** and verified equal by test.

## 12. Recommendation

**No further intervention is warranted on this evidence, and none is recommended.**

Criteria 1-3 passing is interesting; criterion 4 failing on one control while eight of
eleven concepts move is the more informative fact, and it points away from the specific
mechanism the hypothesis proposes. Building a concept-informed adjustment on this would be
building on an effect that has not been shown to be about scarring.

Two things are worth doing, and neither is an intervention:

1. **Resolve the consolidation anomaly (U3)**, or document that it cannot be resolved and
   stop naming these axes after radiological findings. It is now the single largest
   interpretive gap, and it undermines every concept statement in this report and the
   previous one.
2. **If the mechanism question is pursued**, the discriminating experiment is not another
   association comparison. It is to ask whether the *rank order of cases* changes
   specifically among those high on the scarring axis, versus changing everywhere — which
   the current data could answer without any new inference, from the per-case scores that
   already exist, had the concept similarities been retained. They were deliberately not.
   That is a cost of the no-export rule, correctly paid.

**Explicitly not recommended:** opening Vietnam REPORT, opening TBX11K `internal_test`,
adopting B1, score subtraction, apical weighting, or any recalibration based on a concept
score.
