# Unresolved items

Marked unknown rather than inferred from a filename or a summary. Each says what evidence
would settle it.

---

## U1 — Was TBX11K `internal_test` scored on 2026-10-14? **PRIORITY**

`artifacts/p2/sealed/sealed_result_rehearsal1.json` records a scored evaluation:

```
n = 901, n_tb = 101, AUC = 0.992
sens@spec0.70 = 1.000   sens@spec0.95 = 0.9703
operating point: sensitivity 0.9208, specificity 0.9888
triage_counts: TB 102, NOT_TB 799        (= 901)
seal_id "rehearsal1", model_id "radino_full_clahe0_bin", token_issuer "person4"
```

**901 is the size this project records for TBX11K `internal_test`** — see
`realworld_generalization_20261021_1015/tables/cohort_exposure_registry.csv`, which lists
internal_test at n=901 while simultaneously asserting *"NO - never read"*.

Every claim that internal_test is untouched traces back to one **note** in
`artifacts/p2/external/vietnam_eval.json` — *"internal_test untouched."* — not to
evidence. The registry's own evidence column says *"recorded untouched in the
vietnam_eval.json note; no artifact scores it."* An artifact appears to score 901 cases.

**Not concluded, because:** I did not open the split, read its labels or count its rows,
so the partition identity is inferred from size alone. The seal is also named
`rehearsal1`, and the model (`radino_full_clahe0_bin`) is an earlier variant, not the
current regional-MIL configuration — so even if this was internal_test, the current model
has not been evaluated on it.

**What would settle it:** P1 or P4 confirming which case list `rehearsal1` scored, or a
split manifest giving each TBX11K partition's row count. **Confidence: inferred, high.**

**Why it matters:** internal_test is the project's only claimed-pristine partition. If it
has been scored, the project has no unexposed test set at all, and that must be stated in
the paper rather than discovered by a reviewer.

## U2 — Are the nine concept similarity vectors near-collinear?

Scarring and fibrosis are near-synonyms and almost certainly correlate strongly, so the
effective number of independent tests is smaller than 9 and BH over 9 is conservative for
a family that is not independent. It changes no conclusion here (p_BH ≈ 9e-103), but it
should be reported.

**Cannot be computed from what is on disk:** the per-case 1352×9 matrix was deliberately
not exported. **What would settle it:** re-run the TUNE audit exporting a 9×9 correlation
matrix — an aggregate, not a per-case table. Costs one Kaggle run, touches nothing
reserved.

## U3 — Why does consolidation run the wrong way?

ρ = −0.300 among negatives and **−0.385 among positives**. Consolidation is the canonical
active-TB finding; the classifier's score falls as consolidation similarity rises, in the
group that has TB. No explanation is available from the artifacts.

Candidate explanations, none tested: the text direction is dominated by something other
than the radiological finding; negation in the source sentences; the classifier keys on
chronic rather than acute change; a sign or indexing error somewhere in the chain.

**What would settle it:** inspect the highest and lowest consolidation-similarity cases
against the Lunit comparator score (aggregate, no images exported), and check the
similarity of a deliberately orthogonal control concept. **Until resolved, the nine axes
should not be read as radiological findings.**

## U4 — Is CLEAR exposed to this cohort?

The model card names three training corpora and no evaluation set, so overlap with the
Vietnam cohort is **unknown rather than excluded**. If CLEAR has seen these images, the
concept associations are potentially exposed.

**What would settle it:** a statement from the CLEAR authors, or a published list of their
evaluation sets. Not resolvable from repository artifacts.

## U5 — Git history was not searched

This audit covers artifacts on disk. The working copy has 7 modified tracked files pending
from earlier documentation tasks, and `git log` was not traversed. A REPORT read that left
no artifact would be missed by this ledger. Nothing observed suggests one occurred, and
every REPORT event found did leave artifacts.

**What would settle it:** `git log -p --all` filtered for the split file and for the
figures 0.7242 / 0.7448 / 0.0206.

## U6 — Does the current `score_baseline` column correspond to `mil_final.json`?

`vietnam_case_scores.csv` (2,708 rows, from `overnight_scanner_run_20261021_0208`) is the
score file the concept audit consumed. Its `score_baseline` column reproduces the headline
to within 0.0005 per
`realworld_generalization_20261021_1015/00_STATE_AND_EVIDENCE_AUDIT.md:36` (0.7245 vs
0.7240), which is consistent but is a *reproduction*, not an identity check.

**Impact if wrong:** the concept correlations would be against a near-copy of the headline
model rather than the headline model itself. The ρ = 0.566 finding would be essentially
unchanged at that level of agreement. **Low priority.**

## U7 — Did any REPORT read occur before 2026-10-20?

`mil_final.json` is dated 2026-10-20 and its note claims first contact. No earlier
artifact scoring 1,356 Vietnam cases was found. **Confidence that E1 was the first REPORT
read: moderate** — absence of an artifact is weaker evidence than presence of one.
