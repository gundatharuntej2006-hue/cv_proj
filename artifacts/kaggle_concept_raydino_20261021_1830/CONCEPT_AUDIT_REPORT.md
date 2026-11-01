# Stage 1 — Concept audit

## Concept inference did not run. No concept score was computed, and none was invented.

**Blocker:** the Vietnam image licence is unresolved — the Dryad record names no specific
licence — so images may not be uploaded to Kaggle, and per the run's own constraint
*unknown means blocked*. CLEAR itself is Apache-2.0 and would have been usable; the
obstacle is the data, not the model. Notebook `01_concept_audit.ipynb` is gated on an
explicit manifest flag and will refuse to compute concept scores until that flag is set.

**What did run:** the score-only subgroup analysis, which needs no new model — only
committed per-case scores and the Dryad metadata. **Vietnam TUNE only. REPORT was not
read.**

Tables: `tables/subgroup_metrics.csv`, `tables/preflight_audit.json`.

---

## 1. Classification of this evidence

**EXPLORATORY.** Vietnam TUNE has been used repeatedly in earlier experiments and is not
independent validation. The previous-TB subgroup was not prespecified when it was first
examined. Nothing here is a clean generalisation estimate.

## 2. Result: the score elevation in previous-TB negatives is specific to negatives

Frozen baseline score (12×12 max focal pooling), Vietnam TUNE, n = 1,352.

| group | n | median score |
|---|---|---|
| Xpert-negative, **no** previous TB | 796 | **−0.3475** |
| Xpert-negative, **previous TB** | 417 | **+2.0509** |
| Xpert-positive, no previous TB | 90 | +3.1505 |
| Xpert-positive, previous TB | 49 | +3.0858 |

**Among negatives**, previous-TB cases score +2.3984 higher in median
(95% CI [1.9685, 3.1057], 10,000-replicate bootstrap, seed 20261021 — excludes zero), and
the score separates previous-TB from other negatives at **AUROC 0.6957**.

**Among positives, the same separation is at chance: AUROC 0.4973**, with near-identical
medians (+3.086 against +3.151).

## 3. Why that asymmetry matters

This is the finding the brief's analysis 3 asks for, and it cuts two ways.

**Encouraging.** Previous-TB history carries essentially no score signal among true
positives. So a hypothetical suppression keyed on previous-TB *history* would not
differentially penalise previous-TB patients who genuinely have active disease.

**Discouraging, and more important.** Previous-TB negatives sit at +2.05 while all
positives sit near +3.1. Those distributions overlap heavily. **Any suppression strong
enough to push previous-TB negatives down would land squarely in the positive
distribution**, because the model is not separating these groups on a dimension that
distinguishes active from inactive disease. That is a direct argument for caution about
Stage 3, derived from measurement rather than intuition.

## 4. What this does NOT establish

- **It does not show fibrosis, scarring or any image-visible finding.** `past_tb` is a
  **clinical-history** variable from the Dryad release, not a radiologist annotation. No
  radiological concept may be inferred from it.
- **It does not show the classifier uses any particular visual cue.** A score difference
  by history is an association. Establishing that the model relies on a specific finding
  needs concept scores or expert annotation, neither of which is available here.
- **It is not novel.** Poorer CAD performance in people with previous TB is established:
  Kagujje et al. 2022 (*Clin Infect Dis*), Qin et al. 2024 (*Lancet Digit Health*), and
  Codlin et al. 2025 (*PLOS Digit Health*) — the last on **this exact cohort**, for the
  commercial comparator. Human readers degrade on the same axis too.
- **No causal claim** is made or supported.

## 5. The preregistered concept list, fixed and unused

Recorded in `kaggle/01_concept_audit.ipynb` **before** any association could be computed:
fibrosis, scarring, volume loss, calcified granuloma, calcification, cavity, focal
opacity, consolidation, pleural thickening, nodule, reticular opacity, bronchiectasis.

Each must be mapped to an **exact phrase in the official CLEAR concept bank**. Any concept
absent from the released vocabulary is dropped and recorded, never replaced with an
invented phrase. The list is frozen so it cannot be adjusted after seeing associations.

## 6. If the licence is resolved

The notebook would then: encode TUNE images with CLEAR's **official 448 px** preprocessing
at a pinned commit; project to the concept bank; and write
`tables/concept_score_summary.csv` and `tables/concept_model_score_associations.csv` with
bootstrap CIs and **Benjamini–Hochberg correction across the 12 concepts**, stratified by
active-TB label and previous-TB history — **including among positives**, since a concept
that also marks true active disease must not be suppressed.

Per `DECISION_RULES.md`, a credible actionable signal requires **all four** conditions,
and a score-by-history difference alone explicitly does not qualify.

## 7. Limitations

One exposed cohort; a subgroup that was not prespecified; self-reported history that is
not record-verified; no expert annotation; no concept score; and an encoder whose exposure
to this cohort is **unknown**. A concept audit is diagnostic and **is not expected to
change AUROC** — no AUROC change is claimed from it.
