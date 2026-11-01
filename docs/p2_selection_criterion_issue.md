# The D08/D09 selection criterion is degenerate at 70% specificity

**Owner:** Person 2 · **Date:** 2026-10-14 · **Affects:** D08, D09, D11 (not yet locked)
**Status:** found during the first run on real lung crops, before any decision was locked

## The problem

P2-08's acceptance criterion is *"sensitivity at 70% specificity, AUC tie-break"*. On real
TBX11K lung crops, **both CLAHE variants score exactly 1.0000** at that operating point, so
the primary metric cannot distinguish them and the tie-break decides everything silently.

| specificity target | crop_m0.10_clahe0 | crop_m0.10_clahe1 | discriminates? |
|---|---|---|---|
| **0.70** (current criterion) | **1.0000** | **1.0000** | **no — tied** |
| 0.80 | 1.0000 | 0.9831 | yes |
| 0.90 | 0.9831 | 0.9661 | yes |
| 0.95 | 0.9661 | 0.9492 | yes |
| 0.98 | 0.9322 | 0.9153 | yes |
| 0.99 | 0.9322 | 0.8644 | yes |

Two causes, both structural rather than accidental:

1. **The task is too easy at 70% specificity.** A frozen RAD-DINO embedding separates TB
   well enough that every reasonable variant catches every TB case once 30% of negatives
   are allowed through. The criterion was written for the clinical target, not for
   discriminating between models.
2. **The development split has only 59 TB cases**, so sensitivity can only move in steps of
   1/59 = 0.0169. The metric is coarse as well as saturated.

## Why this matters

A pre-registered criterion that always ties is worse than no criterion: it looks rigorous
while the decision is actually made by the tie-break, which was never meant to carry that
weight. Had this not been caught, D08/D09/D11 would have been "decided by sensitivity at
70% specificity" in the report when they were really decided by AUC.

## Recommendation

Separate the two jobs the number is doing:

- **Report at 70% specificity.** That is the WHO screening operating point and must stay
  the headline clinical number.
- **Select at a specificity where the metric discriminates.** Sensitivity at **95%**
  specificity separates the variants cleanly here and is still clinically meaningful (it is
  the stricter end of a triage operating range). AUC remains the tie-break.

Concretely, amend the D08/D09/D11 criterion to *"sensitivity at 95% specificity, AUC
tie-break, reported alongside sensitivity at 70% specificity"*.

**The conclusion does not change.** CLAHE off wins at every specificity from 0.80 to 0.99
and on AUC (0.9940 vs 0.9893), so the preliminary answer is robust to the criterion. This
is about making the stated reason match the real one.

## Process note

This is a change to a pre-registered criterion, so it must be deliberate, dated and
justified by evidence that does not depend on the outcome it selects — which is the case
here: the degeneracy is a property of the metric and the development split size, visible
without reference to which variant wins. The decisions are **not yet locked** (the split is
still provisional pending P1's C01), so amending now costs nothing. Amending after locking
would not be acceptable.

Evidence: `artifacts/p2/dryrun/real_d08_clahe.json`, and the table above, reproduced by the
selection harness at several specificity targets.

---

## Follow-up (2026-10-14): the amendment is only half sufficient

With all ten variants now measured (`p2_d08_d09_recommendation.md`), selecting at 95%
specificity discriminates **D08** cleanly — CLAHE off scores 0.9661 against 0.9492 for CLAHE
on, consistently across all five pairings — but **D09 is a five-way exact tie**: every
non-CLAHE variant, including no crop at all, scores 0.9661, and the AUC tie-break then
decides on 0.0006, which is well inside the bootstrap interval.

So the criterion needs its fallback stated rather than left implicit:

> Select on sensitivity at 95% specificity. Where variants tie on it and the paired bootstrap
> shows no significant AUC difference, prefer the **simpler** pipeline and record that
> simplicity, not accuracy, was the deciding ground.

That keeps the original point of this document — the stated reason must match the real one —
in the case where no metric can separate the candidates at all.
