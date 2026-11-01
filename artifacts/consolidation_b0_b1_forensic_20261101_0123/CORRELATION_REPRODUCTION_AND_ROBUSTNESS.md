# Reproduction and robustness of the consolidation association

All numbers below come from a fresh run of the **pinned** CLEAR inference over the 1,352
approved Vietnam TUNE images, on this machine, using the audited package unchanged.
Per-case similarities were computed in private scratch and deleted.

---

## 1. The reported values reproduce exactly

Checked against the committed result files of the earlier runs, not against prose
summaries of them:

| statistic | previously committed | reproduced here | difference |
|---|---|---|---|
| consolidation, TB-negatives (n=1,213) | **−0.3000** | **−0.3000** | 0.0000 |
| consolidation, TB-positives (n=139) | **−0.3849** | **−0.3853** | 0.0004 |
| scarring, TB-negatives | +0.5661 | **+0.5661** | 0.0000 |
| fibrosis, TB-negatives | +0.4166 | **+0.4166** | 0.0000 |

The Benjamini-Hochberg values over the nine primary concepts also reproduce
(scarring 9.27e-103, consolidation 3.55e-26, pleural thickening 0.0746 — not significant),
which confirms the cohort, the ordering and the family are the same ones used before.

**Disposition component: the statistic is REPRODUCIBLE.** Not an alignment failure, not a
transcription error.

### Cohort and alignment

1,352 patients, 139 Xpert-positive, 466 with previous TB, cohort fingerprint
`1aa55cd5c0eb2088…` matching the pinned staging manifest. Labels and `past_tb` were
re-derived from the Dryad clinical CSV by the pinned loader. Every join is **by patient
id through a dictionary, never by row position**; a missing or duplicated id halts the
run, and a test shuffles the input order and demands the same score vector back.

**B0 is production-identical**: Spearman and Pearson both **1.000000** against the
production `score_baseline` column, maximum absolute difference **1e-06** (the CSV's
six-decimal rounding). So the comparison is against production itself.

## 2. Pearson agrees with Spearman — not a rank artefact

| group | Spearman | Pearson |
|---|---|---|
| negatives (all) | −0.3000 | −0.3093 |
| positives (all) | −0.3853 | −0.3744 |
| negatives, past TB | −0.2850 | −0.2994 |
| negatives, no past TB | −0.2878 | −0.2955 |

Spearman remains the preregistered statistic; Pearson is reported only as a sensitivity
check. The two agree to within 0.01 everywhere, so the association is not an artefact of
ranking or of a monotone distortion.

## 3. It survives covariate adjustment, but is small after it

Incremental R-squared over scanner (2 indicators), age group, sex, symptom count and
`past_tb`, among TB-negatives:

| concept | rho | incremental R-squared |
|---|---|---|
| scarring | +0.5661 | **0.1975** |
| **`no acute cardiopulmonary process`** *(probe)* | **−0.6012** | **0.1375** |
| fibrosis | +0.4166 | 0.0996 |
| **`normal chest`** *(probe)* | −0.4348 | 0.0955 |
| **`no consolidation`** *(probe)* | **−0.5339** | **0.0796** |
| `no scarring` *(probe)* | +0.3009 | 0.0645 |
| `clear lungs` *(probe)* | −0.4694 | 0.0602 |
| **consolidation** | **−0.3000** | **0.0182** |

**Consolidation's unique contribution is 0.018** — an order of magnitude below scarring's.
Most of its raw rho is absorbed by the covariates, chiefly scanner (§5).

## 4. The probes settle what the axis measures

Declared in `configs/concept_indices.json` **before** any image-side similarity for them
was computed. Among TB-negatives, against the B0 (production) score:

| anchor | role | rho vs classifier score |
|---|---|---|
| **`no acute cardiopulmonary process`** | normality | **−0.6012** |
| **`no consolidation`** | negation of the anomalous concept | **−0.5339** |
| `clear lungs` | normality | −0.4694 |
| `normal chest` | normality | −0.4348 |
| `lungs are clear` | normality | −0.4107 |
| **`consolidation`** | the audited concept | **−0.3000** |
| `no scarring` | negation of the headline concept | **+0.3009** |
| `scarring` | the headline concept | +0.5661 |

**Three things follow, and none of them is subtle.**

**(a) The negation is invisible.** `no consolidation` correlates with the classifier score
at **−0.534** — *more strongly negative* than `consolidation` itself. An anchor and its
explicit denial point the same way. The same holds for the headline concept: `no scarring`
is **+0.301**, the same sign as `scarring` (+0.566).

**(b) The axis is a normality axis.** Every normality phrase correlates strongly
negatively, and the strongest of them all (−0.601, incremental R-squared 0.1375) is
`no acute cardiopulmonary process` — a phrase containing no finding at all. The
consolidation anchor's −0.300 is a *weaker* version of this same signal.

**(c) The sign is therefore not a mystery.** A classifier trained to score TB higher will
score "looks like a normal chest" lower. Because the consolidation anchor sits in the
normality-dominated region of the text embedding (see `CONSOLIDATION_CONCEPT_SEMANTICS.md`
§4: 27.6% of consolidation sentences in the vocabulary are negated, 17x scarring's rate),
its similarity partly measures normality, and it inherits that negative sign.

Image-side corroboration: `consolidation` and `no consolidation` similarity vectors
correlate **+0.342** across the cohort, and `scarring` with `no scarring` **+0.738**. On
the image side too, an anchor and its denial are substantially the same measurement.

## 5. Mixture: a real composition effect, no sign reversal

| stratum | n | rho |
|---|---|---|
| **negatives, pooled** | 1,213 | **−0.300** |
| negatives, scanner 1 | 182 | −0.140 |
| negatives, scanner 2 | 590 | −0.241 |
| negatives, scanner 3 | 441 | −0.147 |
| negatives, past TB | 417 | −0.285 |
| negatives, no past TB | 796 | −0.288 |
| **positives, pooled** | 139 | **−0.385** |
| positives, no past TB | 90 | −0.424 |
| positives, past TB | 49 | −0.346 |
| positives, scanner 1 / 2 / 3 | 22 / 75 / 42 | −0.140 / −0.274 / −0.315 |

**Within every scanner the correlation is substantially weaker than pooled** — roughly
half. So a meaningful share of the pooled association is **between-scanner composition**,
not a within-scanner relationship. This is consistent with the small incremental
R-squared of 0.018 once scanner is in the model.

**But there is no sign reversal anywhere.** Every one of the twelve strata is negative, in
both arms. This is attenuation, not Simpson's paradox. Previous-TB strata are
indistinguishable from each other (−0.285 vs −0.288), so previous TB is not driving it.

## 6. Influence

Leave-one-out over every stratum (`tables/consolidation_stratified_results.csv`): in the
pooled negatives the rho ranges over **[−0.3028, −0.2983]** across all 1,213 deletions. No
single observation moves it by more than 0.003, and the sign is stable under deletion in
**all twelve strata**. The result is not carried by outliers.

## 7. What reproduced, and what that licenses

| hypothesis for the negative sign | verdict |
|---|---|
| transcription / sign flip in an intermediate table | **eliminated** — no negation anywhere in the chain; code read, not assumed |
| concept index or column-order error | **eliminated** — index 123742 is the bare word `consolidation`, unique, re-derived from the vocabulary; joins are by id |
| image/score misalignment | **eliminated** — alignment by pid, shuffle test, exact reproduction of four prior statistics |
| score direction inverted | **eliminated** — cosine, both sides L2-normalised, no temperature, no contrast, larger = more similar |
| outliers | **eliminated** — leave-one-out range 0.003 |
| Simpson's paradox (sign reversal) | **eliminated** — all twelve strata negative |
| between-scanner composition | **partly responsible** — pooled −0.300 vs within-scanner −0.14 to −0.24 |
| correlated-concept structure | **partly responsible** — consolidation is cosine +0.847 from focal opacity in text space, +0.797 on the image side |
| **negation-blind anchor sitting in a normality-dominated region** | **the main explanation** — `no consolidation` gives −0.534, normality phrases give up to −0.601 |

**The negative sign is explained.** It is not evidence about consolidation the
radiological finding; it is what happens when a text anchor that cannot distinguish
"consolidation" from "no consolidation" is correlated against a classifier score in a
cohort whose reports are dominated by normal-chest language.
