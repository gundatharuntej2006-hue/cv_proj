# Report consistency review: draft claims against committed artifacts

**No published document was edited by this run.** Three corrections are proposed below as
exact find/replace patches for review. The runbook permits editing a final paper file only
for an objectively false statement; none of these is false, so none was applied.

Checked: `docs/p2_focal_pooling_final.md`, `docs/p2_focal_pooling.md`,
`report/paper_draft.md`, `artifacts/p2/audit/CROSS_DATASET_EVALUATION_REPORT.md`,
`artifacts/p2/audit/RESULT_INTEGRITY_CHECK.md`.

---

## Finding 1 — the 9×9 row of the sweep table has no committed artifact behind it

**Severity: the highest of the three.** This is the one worth acting on.

`docs/p2_focal_pooling_final.md` presents an eight-row sweep table and says *"Eight region
grids, five aggregators, three seeds."* `artifacts/p2/regional/mil_sweep.json` contains
results for grids **1, 2, 3, 4, 6, 12**, plus grid **18** in `grid18_addendum`. **There is no
grid-9 entry anywhere in the artifacts.**

Every other cell in that table was checked against the JSON and matches:

| grid | doc vs artifact |
|---|---|
| 1×1, 2×2, 3×3, 4×4 | all values match exactly |
| 6×6 | mean, max, top3 match; **lse: doc 0.6822, artifact 0.6821** (rounding, trivial) |
| **9×9** | **no artifact entry exists** |
| 12×12 | all four values match exactly (mean 0.6364, max 0.7239, top3 0.7189, lse 0.7229) |
| 18×18 | all four match the addendum exactly (0.6349 / 0.7186 / 0.7224 / 0.7257) |

**Why it happened, as far as the evidence supports.** `scripts/p2_32_mil_sweep.py` line 42
sets `GRIDS = (1, 2, 3, 4, 6, 9, 12, 18)`, and line 157 comments *"18 gives grids 9 and 18,
not poolable from 12"*. So the 9×9 configurations were almost certainly run, in the same
separate pass as grid 18 that followed a memory failure — but only the grid-18 rows were
written into `grid18_addendum`. The 9×9 numbers survive in the document and in lost console
output.

**Why it matters.** The document's interpretation leans on that row: *"The curve climbs
steeply to 9×9, then flattens: 6×6 0.6888, 9×9 0.7205, 12×12 0.7239, 18×18 0.7186."* The
saturation argument is still supported without 9×9 — 6×6 0.6888 → 12×12 0.7239 → 18×18
0.7186 shows the same rise and flattening — but as written, a reader who goes to the
artifact to check the claim will not find it, and the table does not say so.

**Proposed minimal correction** — mark the row rather than delete it, since deleting a
result that was probably obtained would be its own distortion:

```diff
-| 9×9 | 0.6368 | 0.7205 | 0.7110 | 0.7142 |
+| 9×9 \* | 0.6368 | 0.7205 | 0.7110 | 0.7142 |
```

and add directly beneath the table:

```
\* The 9×9 row was produced in the separate pass that followed the memory failure, but
only the 18×18 results from that pass were written to `mil_sweep.json`. These four values
are not reproducible from a committed artifact and should be treated as unverified. The
saturation argument does not depend on them: 6×6 0.6888 → 12×12 0.7239 → 18×18 0.7186
shows the same rise and plateau using verified rows only.
```

and correct the count in the prose:

```diff
-Eight region grids, five aggregators, three seeds, selected on TUNE only.
+Eight region grids, five aggregators, three seeds, selected on TUNE only. Seven of the
+eight grids have committed results; see the note under the table.
```

Re-running grid 9 to restore the artifact would also resolve it, and is cheap relative to
the original sweep — but it would be a new read of TUNE, so the pre-committed selection must
not be revisited on the result.

---

## Finding 2 — the global-pooling baseline on Vietnam is quoted as 0.6744 and no artifact says 0.6744

Three values exist for the same configuration (lungs-only masking, healthy-only negatives,
global pooling, Vietnam full cohort):

| value | source | seeds |
|---|---|---|
| **0.6743** | `artifacts/p2/dryrun/hard_negatives.json` | 5 (17/29/43/61/79) |
| **0.6742** | `artifacts/p2/dryrun/learning_curve.json`, fraction 1.0 | 3 (17/29/43) |
| **0.6744** | `docs/p2_focal_pooling_final.md`, `docs/p2_focal_pooling.md`, and a hardcoded `print` in `scripts/p2_33_mil_final.py` line 127 | — |

The spread is 0.0002, far inside noise, and it changes no conclusion. But 0.6744 appears in
the headline ladder table and matches no saved artifact.

**Proposed correction:** quote **0.6743** throughout, matching the five-seed protocol used
for every other headline number.

```diff
-| + healthy-only negatives, lungs-only masking | 0.6744 |
+| + healthy-only negatives, lungs-only masking | 0.6743 |
```
```diff
-| focal pooling | 0.9889 → 0.9368 | 0.6744 → 0.7242 |
+| focal pooling | 0.9889 → 0.9368 | 0.6743 → 0.7242 |
```

`scripts/p2_33_mil_final.py` line 127 is a console string in a completed experiment script
and is best left alone; changing it would alter a script that produced a committed result.

---

## Finding 3 — the saved cross-dataset predictions mix probabilities and logits

In `artifacts/p2/audit/cross_dataset_case_predictions.csv`:

| column | range | scale |
|---|---|---|
| `score_A_global_pooler` | 0.1079 … 0.9969 | **probability** |
| `score_B_patch_mean_1x1` | −4.7369 … 2.5483 | logit |
| `score_C_regional_max_12x12` | −5.9572 … 7.6954 | logit |

AUROC and AUPRC are rank-based, so **every metric in the cross-dataset report is unaffected**
and all six reproduce exactly (§ below). But the `threshold_from_tbx11k_dev` column in
`cross_dataset_metrics.csv` carries 0.697045 for arm A and −3.336755 for arm C, and nothing
in either file says these live on different scales. A reader comparing thresholds across
arms would draw a wrong conclusion.

**Proposed correction:** add one line to `CROSS_DATASET_EVALUATION_REPORT.md` under *Setup*:

```
Score scales differ by arm: arm A stores a calibrated probability, arms B and C store
logits. AUROC and AUPRC are rank-based and so are unaffected, but the per-arm thresholds
in `cross_dataset_metrics.csv` are not comparable across arms.
```

---

## What was checked and found correct

| claim | verdict |
|---|---|
| Vietnam TUNE 0.7237 / REPORT 0.7242 | **reproduce exactly** from the case-level file |
| Shenzhen A 0.9411 / B 0.8471 / C 0.9690 | **all three reproduce exactly** |
| Montgomery A 0.9289 / B 0.8772 / C 0.9690 | **all three reproduce exactly** |
| Lunit 3.9.0.1 on Vietnam, 0.7827 | **reproduces exactly** |
| selection optimism −0.0005 | confirmed (0.7237 vs 0.7242) |
| "Shenzhen 0.969024, Montgomery 0.968966 — a rounding coincidence" | confirmed |
| arm A is a refitted matched baseline, not the deployed checkpoint | confirmed independently: `learning_curve.json` at fraction 1.0 gives Shenzhen 0.9410 and Montgomery 0.9289 for the same lungs-only/healthy-only global configuration, matching arm A to within rounding, while `bundle/head.npz` is `variant: full_clahe0` |
| 12×12 grid wherever the regional representation is claimed | confirmed: all three regional caches are exactly `(n, 144, 768)` float16 |
| per 1,000 screened at 70% specificity: 61 caught, 43 missed, 269 wrongly referred | arithmetic consistent with REPORT prevalence 0.104 and sensitivity 0.59 at that operating point |
| OOD gate is advisory by default | confirmed in `src/p2_cls/pipeline.py:150` and by 11 passing safety tests |

## One claim that should be added rather than corrected

Neither the draft nor any document reports **per-scanner operating-point behaviour**. The
measurements in `score_distribution_report.md` — twofold referral-rate disparity between
machines at a single threshold, specificity 0.795 against 0.520, and the finding that most of
it is a calibration offset rather than a ranking failure — are new as of this run and are
arguably the most useful operational result the project has. They belong in the paper.
