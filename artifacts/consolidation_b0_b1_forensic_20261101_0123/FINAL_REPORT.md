# Consolidation anomaly and B0/B1 representation audit — final report

**Run:** `artifacts/consolidation_b0_b1_forensic_20261101_0123/`
**Executed:** 2026-11-01, 01:23 -> 02:30 +05:30 (CLEAR inference 1.7 min, analysis 26.4 min)
**Git HEAD before and after:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` — **unchanged**
**Type:** forensic and statistical. **No training. No production change.**

---

## Disposition: **ARTEFACT_OR_PIPELINE_ERROR_IDENTIFIED**

A demonstrable **phrase-mapping and semantic** issue explains the result. The numerical
association is fully reproducible — it is the *interpretation* that was wrong, and the
defect is in the concept anchor, not in the code, the alignment or the arithmetic.

> **The CLEAR concept anchors cannot distinguish a finding from its negation.**
> `no consolidation` correlates with the TB classifier score at **−0.534**, *more
> strongly* than `consolidation` itself (−0.300). `no scarring` correlates at **+0.301**,
> the same sign as `scarring` (+0.566).
>
> **Consolidation is negated in 27.6% of the vocabulary sentences that mention it** — 17
> times scarring's rate — because *"no focal consolidation"* is stock normal-report
> language. Its anchor therefore sits in the normality region of the embedding, and
> normality phrases correlate with the score at up to **−0.601**.
>
> The negative sign is not a fact about consolidation. It is a weak version of
> "this chest reads as normal".

## The seven questions, answered

**1. What exact phrase and score are we calling "consolidation"?**
The bare word **`consolidation`**, concept index **123742**, a unique entry, one of
368,294 free-text MIMIC report sentences. A single phrase — no template, expansion,
pooling or ontology mapping. The score is a plain **cosine similarity** between the
L2-normalised projected image embedding (`model.encode_image`) and the L2-normalised
concept row. No temperature, no contrastive term, no within-image normalisation, **no sign
flip anywhere in the chain** (verified by reading CLEAR's own code, not reconstructing it).

**2. Does the negative association reproduce under the pinned implementation?**
**Yes, exactly.** Re-running the pinned CLEAR inference locally over the 1,352 approved
TUNE images:

| statistic | previously committed | here | diff |
|---|---|---|---|
| consolidation, TB-negatives (n=1,213) | −0.3000 | **−0.3000** | 0.0000 |
| consolidation, TB-positives (n=139) | −0.3849 | **−0.3853** | 0.0004 |
| scarring, TB-negatives | +0.5661 | **+0.5661** | 0.0000 |
| fibrosis, TB-negatives | +0.4166 | **+0.4166** | 0.0000 |

BH values reproduce too (scarring 9.27e-103, consolidation 3.55e-26).

**3. Is the sign due to semantics, mixture, a mapping error, correlated concepts, or
still unknown?**
**Semantics, principally** — the negation-blind anchor sitting in a normality-dominated
corpus region (see the box above). **Mixture contributes**: pooled −0.300 against
within-scanner −0.140 / −0.241 / −0.147, so roughly half the pooled value is
between-scanner composition. **Correlated concepts contribute**: consolidation is cosine
+0.847 from `focal opacity` in text space and +0.797 on the image side.
**Eliminated**: sign flip, index/column-order error, misalignment, score direction,
outliers (leave-one-out range 0.003), and Simpson's paradox — every one of twelve strata
is negative, so there is attenuation but no reversal.

**4. Does it hold within scanner and previous-TB strata?**
**Yes, in all twelve**, in both arms, with the sign stable under leave-one-out
everywhere. Weaker within scanner (−0.14 to −0.24 against −0.300 pooled). Previous TB does
not separate it among negatives (−0.285 with, −0.288 without).

**5. Are the B0/B1 changes broad, or selective?**
**Selective — the previous run's "generalised re-ranking" reading is not supported.**
Spearman between a concept's text-space cosine to the `scarring` anchor and the size of its
shift is **−0.824 (p = 4.9e-05, n = 17)**. All 9 concepts at cosine >= 0.69 moved
negatively (mean −0.0244); of the 8 below, only 3 did (mean +0.0026). **The four normality
anchors — which carry the largest associations in the study, up to −0.601 — did not move
at all** (|delta| <= 0.0042, every interval covering zero).

The earlier reading was a sampling artefact: all nine primary concepts are TB-adjacent and
seven sit inside the scarring neighbourhood, so that set could not tell "this
neighbourhood moved" from "everything moved".

Also: **`degenerative changes` was never a valid negative control** — cosine **0.730** to
scarring, above two *primary* concepts. Its shift is what the relationship predicts.
`scoliosis` (cosine 0.524, image-side −0.005 with scarring) was valid, and did not move.

**6. Do the B1 changes come with changes in positive-case rankings or sensitivity?**
B0 and B1 rank the cohort at Spearman **0.9893** (mean rank shift 3.0% of the cohort), yet
AUROC moves 0.7237 -> 0.7441. All concept changes are **ranking** changes, not calibration
— Spearman is monotone-invariant, so scale shifts cannot produce them. The previously
recorded locked-threshold sensitivity drop (**0.9424 -> 0.9281**, two patients of 139) is
unchanged by this run and remains unresolvable in a 139-positive cohort.

**7. Is there evidence justifying a concept-informed intervention now?**
**No — and the evidence against is stronger than before.** An intervention aimed at
scarring would be aimed at an axis that moves identically for `scarring` (−0.0344) and
`no scarring` (−0.0343). There is no validated axis to intervene on.

## What changed about the project's earlier conclusions

| prior claim | status |
|---|---|
| consolidation rho = −0.300 / −0.385 | **reproduces exactly** |
| scarring rho = +0.566, delta-R2 = 0.198, p_BH ~ 9e-103 | **reproduces exactly** |
| "the consolidation anomaly is unexplained" (U3) | **now explained** |
| "the nine axes should not be read as radiological findings" | **confirmed, and now demonstrated** rather than suspected |
| "8 of 11 concepts moved -> generalised re-ranking" | **count correct, inference wrong.** The movement is localised |
| `degenerative changes` as a negative control | **invalidated** — inside the scarring neighbourhood |
| B1 positive-case safety | **unchanged and still unresolvable** |

**No arithmetic in any prior report was found to be wrong.** Every number checked
reproduced. What needs repair is interpretation — reading concept *names* as radiological
findings.

## Provenance and partitions

18 inputs hashed before use (`tables/provenance_inputs.csv`). The three CLEAR assets match
the pinned record exactly: `best_model.pt` `da1617a7…` (603,060,057 B),
`concept_embeddings_368294.pt` `7241c5b9…` (1,131,400,935 B), `mimic_concepts.csv`
`93e975b7…` (22,208,546 B).

| | |
|---|---|
| cohort | Vietnam TUNE only, **1,352** patients, 139 positive, 466 previous TB |
| fingerprint | `1aa55cd5c0eb2088…` — matches the pinned staging manifest |
| **Vietnam REPORT** | **not read.** 0 REPORT pids in cohort; asserted and tested |
| **TBX11K `internal_test`** | **not read, not referenced** |
| B0 | **production-identical**: Spearman and Pearson 1.000000, max abs diff 1e-06 |
| alignment | by patient id throughout; shuffle test enforces it |
| CLEAR inference | **re-run**, locally, because the prior run deliberately deleted its per-case similarities |
| per-case data | computed in private scratch, **deleted**; no pid, image, clinical row or embedding written here |
| uploads / external contact | **none** |

## 6. Workstream D (image review) — skipped, deliberately

No qualified radiologist is participating, so a visual review could not produce a
ground-truth claim and the brief forbids diagnosing from individual images. More
importantly it was unnecessary: the question it would have addressed — *does this axis
track the finding?* — is answered decisively by the negation probes, which need no image
interpretation at all. Skipping it also avoids opening images for a purpose that could not
have been conclusive.

## 7. Is U3 explained?

**Explained.** Not "partly" — the mechanism is demonstrated twice over, on the text side
(negation cosines +0.75 to +0.98; 27.6% negated corpus context) and independently on the
image side (`no consolidation` at −0.534; four normality anchors at −0.41 to −0.60).

The residual unknowns are not about the sign: the between-scanner composition component is
quantified but not decomposed into site versus population, and CLEAR's exposure to this
cohort (U4) remains unknown.

## 8. Deliverables

| file | contents |
|---|---|
| `CONSOLIDATION_CONCEPT_SEMANTICS.md` | exact phrase, index, formula, direction, negation test, geometry |
| `CORRELATION_REPRODUCTION_AND_ROBUSTNESS.md` | reproduction, Pearson sensitivity, strata, mixture, influence |
| `B0_B1_RERANKING_DECOMPOSITION.md` | family reconciliation, ranking vs scale, the localisation result |
| `DATA_PROVENANCE_AND_PARTITION_CHECK.md` | hashes, cohort, probe declaration, partition guards |
| `LIMITATIONS_AND_CLAIM_BOUNDARIES.md` | what may and may not be said |
| `NEXT_EXPERIMENT_DECISION.md` | **stop the concept-intervention line** |
| `RUN_MANIFEST.json`, `MANIFEST.sha256` | provenance and integrity |
| `tables/` (15 files) | all aggregate; no per-case row anywhere |
| `scripts/`, `tests/` | reproduction and 18 tests |

## 9. Reproduce

```sh
cd D:/tb-triage/artifacts/consolidation_b0_b1_forensic_20261101_0123
py -3 scripts/a1_concept_semantics.py          # text-side only, no images, ~1 min
py -3 scripts/b1_image_side.py                 # dry validation of every guard
py -3 scripts/b1_image_side.py --run-inference # ~27 min, local GPU
cd D:/tb-triage && py -3 -m pytest artifacts/consolidation_b0_b1_forensic_20261101_0123/tests -q
```

Requires the pinned `clear` package
(`pip install --no-deps git+https://github.com/peterhan91/CLEAR.git@57d2eae57ddabf8e56c8013ec675336d71b3d8e7`,
plus `ftfy regex`) and the three CLEAR assets at revision `142f169c…`. Seeds: 20261101,
10,000 replicates for primary groups, 2,000 for subgroups, percentile intervals,
patient-level resampling.

## 10. Production state

**Vietnam screening AUROC remains 0.7242.** Not recomputed — REPORT was not read.

No model was trained or fine-tuned. No head was fitted. No checkpoint, prediction file,
feature cache, threshold, pooling, preprocessing, calibration, tracked document or prior
experiment artifact was created, modified or deleted. No commit, push or deletion. The
audited package `kaggle_exec_prep_20261023_1130` was **imported, never edited** — the
extended concept map was supplied by shadowing its asset lookup at run time with a
temporary file, removed afterwards and asserted gone by test.
