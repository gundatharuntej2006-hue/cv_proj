# Expected outputs

Six files. Cell 6 of the notebook validates every one against the rules below and raises
rather than letting a malformed or per-case output through. **Nothing here exists yet** —
these are the shapes a successful run must produce.

Output root: `/kaggle/working/` on Kaggle, the package directory locally.

---

## 1. `tables/concept_associations.csv` — the primary table

**36 rows** = 9 concepts × 4 groups. Columns, alphabetically as written:

| column | type | meaning |
|---|---|---|
| `ci95_hi`, `ci95_lo` | float | percentile bootstrap interval for `spearman_rho`, 10,000 case-level resamples, seed 20261023 |
| `concept` | str | one of the 9 exact vocabulary entries |
| `group` | str | `negatives (all)`, `negatives, past TB`, `negatives, no past TB`, `positives (all)` |
| `n` | int | cases in the group |
| `p_bh_adjusted` | float or empty | **only for `negatives (all)`.** Benjamini–Hochberg across the 9 concepts. Empty elsewhere, because those groups are not the primary family |
| `p_raw` | float | uncorrected Spearman p |
| `spearman_rho` | float in [-1, 1] | concept similarity vs the frozen classifier score |

Group sizes are fixed by the cohort and must come out exactly:

| group | n |
|---|---|
| `negatives (all)` | **1213** |
| `negatives, no past TB` | **796** |
| `negatives, past TB` | **417** |
| `positives (all)` | **139** |

Validation rules: 36 rows; `ci95_lo <= spearman_rho <= ci95_hi`; `p_bh_adjusted >= p_raw`;
exactly 9 non-empty `p_bh_adjusted` values; no group with `n < 20`.

## 2. `tables/concept_incremental_r2.csv`

**9 rows**, one per concept. Columns: `concept`, `r2_clinical_only`, `r2_with_concept`,
`delta_r2`.

`r2_clinical_only` is identical on every row — it is the TB-negative-only regression of
the classifier score on scanner (2 dummies), age group, sex, symptom count and previous
TB. `delta_r2 = r2_with_concept - r2_clinical_only` and is `>= 0` up to rounding, because
the concept is added to that same model. **In-sample R², not cross-validated**, so a small
positive `delta_r2` is expected even for noise; the noise case is bounded at under 0.02 by
test.

## 3. `tables/concept_distributions.csv`

**36 rows** = 9 concepts × 4 label-by-history groups. Columns: `concept`, `group`, `n`,
`mean`, `sd`, `p25`, `median`, `p75`, `note`.

Groups: `negatives, no past TB` (796), `negatives, past TB` (417),
`positives, no past TB` (**90**), `positives, past TB` (**49**).

A group with `n < 20` is written with `note = "suppressed, n < 20"` and empty statistics
rather than a number nobody should read. None of the four should trigger it at these
sizes; the rule is there so a re-cut cohort cannot quietly produce a statistic from 6
cases.

## 4. `tables/concept_discrimination.csv`

**9 rows**. Columns: `concept`, `n_pos`, `n_neg`, `auroc`, `auroc_ci95_lo`,
`auroc_ci95_hi`, `auprc`, `prevalence`, `note`.

`n_pos = 139`, `n_neg = 1213`, `prevalence = 0.1028`. 2,000 bootstrap resamples.

**This is descriptive only.** It asks how well a single CLEAR text-similarity separates
Xpert-positive from Xpert-negative participants. A concept score is **not** a TB
classifier, was not trained for this, and a high value here would not be a screening
result; `note` carries that caveat into the file so it travels with the numbers. If either
class fell below 20, the row is written with `note` beginning `suppressed` and no metrics.

## 5. `tables/preflight_kaggle.json`

The full preflight record: runtime, dependency versions against the pinned ones, resolved
model revision and asset checksums, cohort fingerprint, expected similarity shape,
partition scan, warnings, blockers, verdict. `verdict` must be `GO`; cell 2 halts
otherwise.

## 6. `run_manifest.json`

Provenance for the run. Shape and meaning: `RUN_MANIFEST_TEMPLATE.json`. Must contain:

- `data.cohort_fingerprint_sha256` = `1aa55cd5c0eb2088f2a037d9a8e5770ca1c0f4a75ea21566d743d17577ed1751`
- `data.n_cases` = 1352, `data.report_partition_used` = `false`,
  `data.tbx11k_internal_test_used` = `false`
- `model.revision` = `142f169c8b006b0f28cf55c6a237c3003db5ed8f`
- `inference.loaded_fraction` **>= 0.90** — below that the run aborts instead of reporting
- `code_sha256` for all six code and config files
- `concepts.score_kind`, which states in the file itself that a concept score is a weakly
  supervised proxy and not a validated detector

---

## Intermediate files — not outputs

`_shards/shard_??????.npy` and `_shards/progress.json` exist only to make cell 4 resumable.
They hold **per-case** similarities. Leave them in the session: do not download them, do
not copy them into the repository, and do not derive a per-case table from them.

## The shape that matters

The similarity matrix is **(1352, 9)**, rows in load order, and the loader's ordered id
hash is what a resume is keyed to. `concept_audit.main` and notebook cell 4 both assert
this shape before any analysis runs, because a silent off-by-one between ids and rows
would produce a table that looks fine and means nothing.
