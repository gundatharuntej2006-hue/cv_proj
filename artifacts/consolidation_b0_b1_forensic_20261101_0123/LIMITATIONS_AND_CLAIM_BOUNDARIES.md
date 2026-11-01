# Limitations and claim boundaries

## 1. What may be said

- **The consolidation statistic reproduces exactly** from the declared inputs and pinned
  method: −0.3000 among TB-negatives, −0.3853 among TB-positives.
- **The CLEAR anchors used in this project cannot distinguish a finding from its
  negation.** `volume loss` and `no volume loss` are cosine +0.980 apart in text space;
  `scarring` and `no scarring` +0.906. On the image side, `no consolidation` correlates
  with the classifier score at −0.534 and `no scarring` at +0.301 — the same signs as
  their affirmative counterparts.
- **The negative sign of the consolidation association is explained** by that fact
  combined with the corpus: 27.6% of the vocabulary sentences mentioning consolidation are
  negated, 17 times scarring's rate, because *"no focal consolidation"* is stock
  normal-report language. Normality anchors correlate with the score at up to −0.601.
- **The B0 -> B1 change in concept associations is localised**, not general: Spearman
  −0.824 (p = 4.9e-05) between a concept's text-space cosine to the scarring anchor and
  the size of its shift; four normality anchors do not move.
- **No production artifact changed.** Vietnam screening AUROC remains 0.7242.

## 2. What may NOT be said

- **Not** that a CLEAR concept similarity is a radiologist-confirmed finding. It is
  cosine similarity to one short sentence from a MIMIC report corpus. There are **no
  radiologist annotations for this cohort**, so no concept statement here has been checked
  against ground truth at all.
- **Not** that the classifier "uses", "relies on", "responds to" or "stopped using" any
  visual feature. Every statistic here is an observational correlation between a model
  output and a text-similarity proxy. Causation is not established and was not tested.
- **Not** that consolidation is absent in high-scoring images. The axis does not measure
  presence or absence; that is the finding.
- **Not** that the scarring result is invalid. Its arithmetic reproduces exactly
  (rho 0.5661, p_BH 9.27e-103). What is invalid is reading the *name*: the axis behaves
  like a topic region containing both `scarring` and `no scarring`.
- **Not** that B1 is safe, beneficial or ready for adoption. Localisation is not safety.
  The previously recorded locked-threshold sensitivity drop (0.9424 -> 0.9281, two patients
  of 139) is unchanged and remains unresolvable in this cohort.
- **Not** independent validation of anything. Vietnam TUNE selected the pooling
  configuration and generated the hypothesis; it is exposed, and nothing measured on it is
  external.
- **Not** a claim of novelty or clinical readiness.

## 3. Specific limitations of this run

**The six diagnostic probes were chosen after the anomaly was known.** They were selected
from the text-side geometry — no patient data, no classifier score — and declared in
writing in `configs/concept_indices.json` before any image-side similarity for them was
computed, which is the strongest guarantee available after the fact. It is **not** the same
as having preregistered them before the anomaly existed. They are diagnostic only and are
excluded from the BH family.

**The §3 localisation analysis in `B0_B1_RERANKING_DECOMPOSITION.md` is post-hoc.** The
relationship between a concept's distance from the scarring anchor and its shift was not
predicted in advance. n = 17 points, of which 6 are probes selected for being far from
scarring, so the range of the predictor is partly constructed. The p-value should be read
as descriptive.

**The BH family is not independent.** Nine TB-adjacent concepts with text-side cosines up
to +0.875 are not nine hypotheses. The correction is conservative for a correlated family,
not a correction for nine independent tests. This is now measured
(`tables/concept_similarity_matrix.csv`) rather than assumed.

**The negation test is limited to pairs that exist in the vocabulary.** Nine such pairs
were found across eleven concepts; `bronchiectasis`, `calcified granuloma`, `scoliosis` and
`degenerative changes` have no exact negated entry, so they were not tested. No phrase was
invented to fill a gap.

**Covariate adjustment is in-sample and observational.** Incremental R-squared is not
cross-validated, so small positive values are expected even for noise, and the covariate
set (scanner, age group, sex, symptom count, `past_tb`) is the one declared by earlier
runs — not a complete confounder model.

**`past_tb` is self-reported clinical history**, not a radiographic annotation of fibrosis
or scarring. It never was, and nothing here changes that.

**Subgroup cells are small.** Positives split by scanner are 22, 75 and 42; by previous TB
are 90 and 49. All subgroup analyses are exploratory and unadjusted, and their intervals
overlap heavily.

**Between-scanner composition is quantified but not resolved.** Pooled −0.300 against
within-scanner −0.14 to −0.24 shows composition contributes, but the three scanners are
confounded with site and population, and this design cannot separate them.

**CLEAR's own exposure to this cohort is unknown.** Its model card names three training
corpora and no evaluation set. If CLEAR has seen these images, every concept association
in this project is potentially exposed. Unresolvable from any artifact here (item U4).

**No image review was performed.** Workstream D was skipped deliberately — see
`FINAL_REPORT.md` §6. No qualified radiologist is participating, so a visual review could
not have produced a ground-truth claim, and the text-side and probe evidence already
settle the question it would have addressed.

**Cavity, nodule and reticular opacity remain unmeasured.** No exact vocabulary entry
exists and none was approximated. Cavity is the finding that most distinguishes active
from healed TB, so the concept set is still missing the variable most relevant to the
clinical question.

## 4. The strongest allowable wording

> Similarity to a short radiological text phrase is associated with the model score.

And for this project's prior reports specifically:

> The nine "concepts" are topic anchors in a report-language embedding, not finding
> detectors. Their names should not be read as radiological findings, and an anchor and
> its explicit negation occupy nearly the same place.
