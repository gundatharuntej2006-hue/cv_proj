# Why the B0 -> B1 concept associations moved

**Headline: the previous run's reading was wrong, and this run can say why.**

`b1_clear_mechanism_20261031_2328` found 8 of 11 concepts shifting with intervals
excluding zero and concluded **"a generalised re-ranking of the score, not one axis being
specifically removed."** With six anchors that sit *outside* the scarring neighbourhood,
that conclusion does not survive: **the four normality anchors do not move at all**, and
the size of each concept's shift is predicted by its distance from the scarring anchor.

The effect is **localised**, not general.

---

## 1. Family reconciliation — 9, 11, 17

| run | encoded | composition |
|---|---|---|
| `concept_audit_result_20261024_0130` | **9** | 9 primary (cavity, nodule, reticular opacity excluded — no exact vocabulary entry) |
| `b1_clear_mechanism_20261031_2328` | **11** | the same 9 + 2 negative controls (`scoliosis`, `degenerative changes`) |
| **this run** | **17** | the same 11 + 6 diagnostic probes |

The nine primary indices are **byte-identical across all three** (test enforced). Nothing
was redefined, dropped or renumbered; two controls were added for the second run and six
probes for this one. The BH family has been the same 9 primary concepts throughout, and
controls and probes have never entered it.

"8 of 11" was therefore an accurate count of *that run's* encoded set — 7 primary and 1
control — but it was an uninformative denominator, because 7 of the 9 primary concepts sit
inside the scarring neighbourhood (§3).

## 2. Ranking versus scale

| quantity | value |
|---|---|
| B0 vs B1 Spearman (rank agreement) | **0.9893** |
| B0 vs B1 Pearson | 0.9899 |
| mean rank shift, as a fraction of the cohort | **0.0302** |
| max rank shift | 0.2241 |
| B0 AUROC / AUPRC | 0.7237 / 0.2007 |
| B1 AUROC / AUPRC | **0.7441** / **0.2241** |

The two arms rank the cohort almost identically — the average patient moves 3% of the way
through the ordering — yet AUROC moves +0.0204. Small rank changes, concentrated where
they matter for discrimination.

**Scale changes are irrelevant to every concept statistic reported here, by construction.**
Spearman is invariant to any monotone transformation, so recalibrating B1 onto B0's
marginal distribution leaves the concept correlations unchanged (verified: consolidation
−0.2933, scarring +0.5317, identical to the raw values). That check is a tautology for
Spearman and is reported as such, **not** as evidence. Its only informative content is
that Pearson (0.9899) and Spearman (0.9893) agree, so there is no large nonlinear
distortion hiding between the two arms.

**Therefore every concept-association change reported is a ranking change.** It is not
calibration, not a level shift, not a prevalence or denominator artefact. (The *absolute*
score levels do differ substantially — B1's negatives sit about 1.7 logits lower — which
is why the two arms have very different locked thresholds. That difference is invisible to
Spearman.)

## 3. The movement is predicted by distance from the scarring anchor

For each of the 17 encoded concepts, its text-space cosine to the `scarring` anchor
against its paired change in association (`tables/delta_vs_scarring_proximity.csv`):

| concept | cos to `scarring` | delta rho | excludes 0 | role |
|---|---|---|---|---|
| scarring | 1.000 | **−0.0344** | yes | primary |
| **`no scarring`** | 0.906 | **−0.0343** | yes | **probe** |
| pleural thickening | 0.875 | −0.0350 | yes | primary |
| calcified granuloma | 0.809 | −0.0077 | no | primary |
| calcification | 0.809 | −0.0297 | yes | primary |
| fibrosis | 0.746 | −0.0182 | yes | primary |
| **`degenerative changes`** | 0.730 | −0.0384 | yes | **control** |
| focal opacity | 0.726 | −0.0106 | yes | primary |
| volume loss | 0.691 | −0.0111 | yes | primary |
| `clear lungs` | 0.654 | **+0.0009** | **no** | probe |
| `no acute cardiopulmonary process` | 0.653 | **+0.0018** | **no** | probe |
| `lungs are clear` | 0.635 | **−0.0020** | **no** | probe |
| `no consolidation` | 0.624 | **−0.0019** | **no** | probe |
| bronchiectasis | 0.587 | +0.0177 | yes | primary |
| **consolidation** | 0.587 | **+0.0067** | **no** | primary |
| `scoliosis` | 0.524 | −0.0070 | no | control |
| `normal chest` | 0.465 | +0.0042 | no | probe |

> **Spearman(cosine to scarring, delta rho) = −0.824, p = 4.9e-05, n = 17.**
>
> Of the **9** concepts at cosine >= 0.69, **9 of 9** moved negatively, mean **−0.0244**.
> Of the **8** below 0.69, only 3 moved negatively, mean **+0.0026**.

**The four normality anchors are the decisive observation.** They carry the largest
associations in the whole study — `no acute cardiopulmonary process` at rho −0.601 with
incremental R-squared 0.1375 — so they span the dominant direction of variance in the
score. If B1 had broadly re-ranked the cohort, these would have moved most of all.
**Their paired intervals all cover zero** (|delta| <= 0.0042).

## 4. Two corrections to the previous run

**(a) "Broad re-ranking" was a sampling artefact of the concept set.** All nine primary
concepts are TB-adjacent, and seven of them sit at cosine >= 0.69 to scarring. A set that
is mostly one neighbourhood cannot distinguish "this neighbourhood moved" from "everything
moved". Adding anchors from outside the neighbourhood resolves it immediately.

**(b) `degenerative changes` was never a valid negative control.** Its cosine to the
scarring anchor is **0.730** — above `focal opacity` and `volume loss`, both of which are
*primary* concepts. It sits inside the region that moved, so its shift (−0.0384) is
exactly what the §3 relationship predicts. The previous run flagged its correlation with
scarring (+0.246 image-side) as a weakness; the text geometry shows it was worse than a
weakness.

`scoliosis`, at cosine 0.524 and image-side correlation −0.005 with scarring, was and
remains a valid control. **It did not move** (−0.0070, interval covering zero).

**Consequence for the previous verdict.** Criterion 4 of that preregistration was recorded
as failing because `degenerative changes` moved. On this evidence it failed because the
control was misspecified, not because the effect was non-specific. That does **not**
retroactively turn the previous verdict into mechanistic support — a preregistered
criterion cannot be re-scored after the fact with a control chosen later, and this run was
not preregistered as an arbiter of it. It does mean the "generalised re-ranking"
*explanation* for that failure is not supported.

## 5. What the localisation does and does not mean

**Supported:** B1 changed the classifier's rank-ordering specifically with respect to a
**topic region** of concept space — the region around the `scarring` anchor — while leaving
its relationship to the normal/abnormal axis essentially untouched.

**Not supported, and this is the limit that matters:** that B1 changed the model's use of
*scarring the radiological finding*. The anchor `no scarring` moved by **−0.0343**, within
0.0001 of `scarring`'s −0.0344. An intervention that genuinely reduced reliance on visible
scarring would not move an anchor that denies scarring by an identical amount. What moved
is a **neighbourhood of report language**, and that neighbourhood contains a finding and
its negation in the same place.

Consistent with the training change: B1 relabels `latent_tb` (obsolete/healed TB) from
positive to negative, which is precisely an instruction about the chronic-appearance topic
region — and says nothing about normality in general.

## 6. Clinical-group decomposition

Consolidation association by group, under B0 (`tables/b0_b1_concept_associations.csv`):

| group | n | rho |
|---|---|---|
| negatives, no past TB | 796 | −0.2878 |
| negatives, past TB | 417 | −0.2850 |
| positives, no past TB | 90 | −0.4236 |
| positives, past TB | 49 | −0.3459 |

Previous TB does not separate the groups among negatives (−0.288 vs −0.285). Among
positives the cells are small (90 and 49); the intervals overlap heavily and nothing
should be read into the difference.

The previous run's finding that **scarring's** reduction concentrates among previous-TB
negatives (−0.0512) relative to others (−0.0274) stands — but §5 constrains how it may be
read, since `no scarring` behaves the same way.

**On the positive-case safety question**: the previous run recorded locked-threshold
sensitivity falling 0.9424 -> 0.9281, two patients of 139. Nothing in this run changes
that number or makes it more resolvable. A concept-association change that is localised
rather than general does not make a two-patient sensitivity loss safe, and the cohort
still cannot measure it.

## 7. Language

Throughout this document and its tables: **"the association changed"** and **"the
score-concept relationship changed"**. Not "the model stopped using", not "the feature was
removed", not "the shortcut was mitigated". No observational correlation here licenses a
claim about what the classifier internally uses.
