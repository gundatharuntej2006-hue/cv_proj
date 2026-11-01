# Integrity checks — DA/DB focal-pooling evaluation

Machine-readable record: `tables/metrics.json` → `integrity`,
`tables/experiment_manifest.json`.

---

## 1. Streaming equivalence, proved before bulk use

Host RAM was 2.44 GB of 16.4 GB while the TBX11K regional cache is 1.35 GB, so features
were streamed from the npz in blocks straight to the GPU. That substitution is legitimate
only if it computes the same thing as the committed path, so it was checked on a fixed
64-case sample against `p2_32_mil_sweep.repool(X, 12, 12)`:

| | |
|---|---|
| shape / dtype | (64, 144, 768) float16, both paths |
| **elementwise identical** | **100.000000%** |
| max absolute difference | **0.0** |
| mean absolute difference | 0.0 |

## 2. End-to-end equivalence against the published arms

The stronger check: do these arms reproduce the numbers `p2_36_cross_dataset_pooling.py`
published for Shenzhen and Montgomery?

| quantity | published | obtained |
|---|---|---|
| arm A `global_pooler`, TBX11K dev AUROC | 0.9570 | **0.9570** |
| arm B `regional_max_12x12`, TBX11K dev AUROC | 0.9368 | **0.9368** |
| arm A threshold at 70% spec on TBX11K dev | 0.697045 | **0.69705** |
| arm B threshold at 70% spec on TBX11K dev | −3.336755 | **−3.33676** |

The script refuses to score DA/DB if either dev AUROC misses by more than 0.01.

## 3. Shape guards

The previous 18×18-against-12×12 mismatch would not have raised an error — a per-region
linear head trains on 144 and scores 324 silently. Three explicit assertions now stand
between that and a result:

| assertion | value |
|---|---|
| `p2_31_regional_mil.GRID` | **12** |
| RAD-DINO patch grid reported by the embedder | **37** |
| extracted array shape | **(176, 144, 768)** float16 |
| id ordering equals the source manifest ordering | **True** |

All three are re-asserted in `tests/test_dadb_focal.py`.

## 4. Case alignment and label derivation

The DA/DB **global** embedding stores no `image_id` — only `emb` (176, 768). Alignment is
therefore positional, through `sorted(SRC.rglob("*"))`, exactly as
`scripts/p2_26_dadb_third_site.py::images()` defines it. The proof that this is safe:

- the **regional** cache written by this run *does* store `image_id`, and its ordering was
  asserted equal to the source-manifest ordering;
- labels and sites were **re-derived from the filesystem** independently of
  `predictions.csv` and match on all 176 cases;
- the per-site composition recomputed from the filesystem matches the published
  `dadb_third_site.json` exactly: **DA 72/22, DB 104/52, pooled 176/74**.

| check | result |
|---|---|
| cases | 176 |
| unique case IDs | 176, **zero duplicates** |
| labels re-derived from filename match stored labels | **True, all 176** |
| sites re-derived from file suffix match stored sites | **True, all 176** |
| published per-site counts match | **True** |
| all three arms scored the same cases in the same order | **True** |

Patient identifiers are not distributed with DA/DB, so patient-level overlap cannot be
checked. Stated, not assumed away.

## 5. Score-array checks

| arm | non-finite | unique values | range |
|---|---|---|---|
| `A_primary_5seed` | 0 | 176 | probability scale |
| `A_secondary_seed17` | 0 | 176 | probability scale |
| `B_regional_max_12x12` | 0 | 176 | logit scale |

**No two arms share a score array** — all three pairwise comparisons return False. Note
that arm A stores calibrated probabilities and arm B logits; AUROC and AUPRC are
rank-based so every metric is unaffected, but the two arms' thresholds are on different
scales and must not be compared to each other.

## 6. Metric recomputation

Every AUROC and AUPRC in `tables/metrics.json` was computed with `sklearn.metrics`
(`roc_auc_score`, `average_precision_score`) directly from `tables/predictions.csv`, not
from the project's `selection.auc`. A test independently recomputes the pooled AUROC for
all three arms from the saved CSV and requires agreement to **1e-9**.

The paired bootstrap distribution is saved in full (`tables/paired_bootstrap.csv`, 4,000
rows) and a test re-derives the reported interval from it, requiring agreement to **1e-9**.

## 7. The reproduced published number

The published DA/DB lungs-only AUROC of 0.7791 looked inconsistent with arm A's 0.9172. It
is not a discrepancy, and the cause was verified by reproduction rather than asserted:
refitting the global head with `p2_26`'s training rows (**all** negatives, 5,941 rows,
no category filter — confirmed by reading `p2_26_dadb_third_site.py:146-160`) gives

```
DA+DB 0.7791   DA 0.8691   DB 0.7973
```

**identical to the published values to four decimals.** The matched arm A used here uses
healthy-only negatives, the current configuration. Recorded in
`tables/negative_composition_check.json`.

## 8. No DA/DB information entered any fitting decision

| | |
|---|---|
| DA/DB labels used to fit any head | **No** |
| DA/DB labels used to set any threshold | **No** — both thresholds come from TBX11K development |
| DA/DB used to select a seed, preprocessing or checkpoint | **No** |
| Vietnam offset correction applied | **No** |
| RAD-DINO retrained or fine-tuned | **No** |
| 18-configuration pooling sweep re-run | **No** |
| manifest revised after seeing a DA/DB metric | **No** — written by `score_dadb.py` before any metric existed; `compute_metrics.py` only reads it |

The feature-extraction script reads no label at all (`labels_used: false` in
`tables/regional_extraction.json`).

## 9. Tests

Full captured output in `test_results.txt`.

| suite | result |
|---|---|
| run-directory tests (`tests/test_dadb_focal.py`) | **19 passed**, exit 0 |
| full production suite, unmodified | **228 passed, 1 skipped**, exit 0 |
| `py_compile` on every script and test written this run | exit 0 |

The 19 tests cover: cohort composition, arm-order alignment, label and site re-derivation,
regional id ordering, the 12×12 shape guard, the committed `GRID` constant, the no-tuning
declarations, threshold provenance against `p2_36`'s recorded values, streaming
equivalence, metric recomputation from the saved CSV, duplicate and non-finite checks,
score-array distinctness, bootstrap-interval reproduction, the label-blind identification
of the sanity failure, inclusion counts, invariance of the conclusion to that exclusion,
and an upstream-hash check that nothing outside this run directory changed.

## 10. Hashes

| artifact | sha256 (first 16) |
|---|---|
| `tables/regional_dadb_lungs_g12_v1.npz` | `8be1b4f90626e045` |
| `scripts/common.py` | `d1a654e9c7734dc2` |

Full list, including every upstream input, in `tables/experiment_manifest.json` → `files`.
`tests/test_dadb_focal.py::test_upstream_artifacts_are_unchanged` re-hashes all of them and
fails if any differs.
