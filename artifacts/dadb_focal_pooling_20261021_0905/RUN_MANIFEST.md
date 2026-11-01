# Run manifest — DA/DB focal-pooling evaluation, 2026-10-21

**Directory:** `artifacts/dadb_focal_pooling_20261021_0905/`
**Start 09:05 · end 09:33 local · elapsed ≈28 minutes**
**HEAD at start and end:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, branch `master`
**Working tree at start:** clean apart from three untracked run directories

Machine-readable manifest, written by `scripts/score_dadb.py` **before any DA/DB metric
existed**: `tables/experiment_manifest.json`.

---

## 1. Environment

| | |
|---|---|
| Python | 3.13.2 |
| torch | 2.6.0+cu124 · CUDA on NVIDIA GeForce RTX 4050 Laptop |
| numpy 2.2.6 · scikit-learn 1.9.0 · scipy 1.17.1 · pandas 3.0.3 | |
| host RAM at start | 2.44 GB available of 16.39 GB (85% used) |
| VRAM | 5.32 GB free of 6.44 GB; 80% cap 5.15 GB; **peak used 0.90 GB** |

## 2. Configuration

| | |
|---|---|
| backbone | `microsoft/rad-dino` ViT-B/14, **frozen**, revision `110cbc18d5133582e320b43d53bf5c44e410c936`, 518 px, patch grid 37 |
| retrained or fine-tuned | **no** |
| preprocessing | lungs-only: resize 512 INTER_AREA if needed, mask from `data/processed/v1/masks_dadb`, non-lung filled with the image mean |
| region grid | **12 × 12 = 144**, asserted three ways |
| training data | TBX11K train, **healthy-only negatives**, 3,241 rows (541 TB) |
| early stopping | TBX11K development, 659 rows (59 TB) |
| seeds | **17, 29, 43, 61, 79** |
| target specificity for thresholds | 0.70, on TBX11K development only |
| bootstrap | 4,000 resamples, seed 0, paired on identical cases |

### Arms

| arm | representation | aggregation |
|---|---|---|
| **A** `global_pooler` | RAD-DINO `pooler_output` | none |
| **B** `regional_max_12x12` | 12×12 region logits | max |

**PRIMARY** — both arms use the 5-seed ensemble (matched seed treatment).
**SECONDARY** — arm A uses seed 17 alone, replicating `p2_36_cross_dataset_pooling.py`,
which took `res[0]` for its global arm. Declared before scoring so the DA/DB numbers are
comparable with the published Shenzhen and Montgomery ones. The two differ by ≤0.0012.

### Frozen thresholds, from TBX11K development only

| arm | threshold | `p2_36` recorded |
|---|---|---|
| A primary (5-seed) | 0.69329 | — |
| A secondary (seed 17) | 0.69705 | 0.697045 ✓ |
| B regional max | −3.33676 | −3.336755 ✓ |

## 3. Cohort

176 files under `data/raw/tbx11k/TBX11K/imgs/extra/da+db`, ordered by
`sorted(SRC.rglob("*"))` — the ordering `scripts/p2_26_dadb_third_site.py::images()`
defines, and the only alignment key the global embedding supports.

| | n | TB | non-TB | prevalence |
|---|---|---|---|---|
| DA (`.png`) | 72 | 22 | 50 | 0.306 |
| DB (`.jpg`) | 104 | 52 | 52 | 0.500 |
| pooled | **176** | **74** | 102 | 0.420 |

Matches `dadb_third_site.json` exactly. Zero duplicate IDs, zero missing files, zero
overlap with any training / development / calibration / internal-test ID. Labels from the
filename first character (`p` → 1), re-derived from the filesystem and matching on all 176.

Cohort manifest sha256 recorded in `tables/experiment_manifest.json`.

## 4. Inclusion rule, fixed before scoring

**Primary: all 176.** The committed pipeline scored all 176 — `p2_26` increments a
sanity-failure counter but excludes nothing — so the primary analysis matches what the
project actually did.

**Sensitivity: 175, excluding `px4`.** Basis: the pre-existing **label-blind** validity
rule `p1_build_c04_c05.roi_from_mask → sanity_pass`. `px4` fails on `components` — one
connected lung field ≥1% of the frame where two are expected (area fraction 0.1056, bbox
(211, 75)–(419, 257), i.e. one lung missed by the segmenter). The rule takes the mask and
nothing else; a test asserts its signature is `(m)` so it cannot read a label. The case was
identified from geometry, and **no case was excluded because it changed a metric**.

Both analyses are reported in full.

## 5. No-tuning guarantee

| | |
|---|---|
| DA/DB labels used to fit any head | **No** |
| DA/DB labels used to set any threshold | **No** |
| DA/DB used to select a seed, preprocessing variant or checkpoint | **No** |
| Vietnam offset correction applied to DA/DB | **No** — explicitly out of scope |
| RAD-DINO retrained or fine-tuned | **No** |
| 18-configuration pooling sweep re-run | **No** |
| `bundle/head.npz` (`variant: full_clahe0`) used as the global arm | **No** — arm A is refitted under the matched configuration |
| manifest revised after any DA/DB metric | **No** |

Feature extraction reads no label at all (`labels_used: false` in
`tables/regional_extraction.json`).

## 6. Order of operations

1. Read-only Phase 0: git state, protocol, cohort inventory, provenance, sanity-failure
   identification (label-blind), configuration verification.
2. Extract DA/DB regional 12×12 features to the **run directory**, asserting `GRID == 12`,
   patch grid 37, shape (176, 144, 768) and id ordering. No existing cache touched.
3. Prove the streaming loader equivalent to the committed `repool` on a fixed 64-case
   sample: **100.000000% identical, max abs diff 0.0**.
4. Fit both arms on TBX11K; require reproduction of the published dev AUROCs 0.9570 and
   0.9368 before continuing.
5. Freeze thresholds, write `tables/experiment_manifest.json`.
6. **Score DA/DB once.**
7. Metrics and integrity checks from the saved per-case file only.

## 7. Deliverables

| file | contents |
|---|---|
| `RUN_MANIFEST.md` | this file |
| `RESULTS.md` | headline results, per-cohort tables, interpretation |
| `INTEGRITY_CHECKS.md` | equivalence, alignment, recomputation, hashes, tests |
| `LIMITATIONS.md` | what cannot be concluded |
| `tables/experiment_manifest.json` | frozen protocol, hashes, seeds, guarantees |
| `tables/case_inventory.csv` | 176 rows: id, site, label, sanity verdict, inclusion, rationale |
| `tables/predictions.csv` | 528 rows (176 × 3 arms): id, cohort, label, arm, score, inclusion flags |
| `tables/metrics.json` | full-precision metrics, CIs, integrity block, seed sensitivity |
| `tables/paired_bootstrap.csv` | 4,000 resampled pooled deltas |
| `tables/regional_dadb_lungs_g12_v1.npz` | the new 12×12 features, with `image_id` |
| `tables/regional_extraction.json` | extraction metadata and hash |
| `tables/negative_composition_check.json` | reproduction of the published 0.7791 |
| `tables/heads.npz` | fitted head weights for both arms, all five seeds |
| `test_results.txt` | captured output of all suites |
| `scripts/`, `tests/` | reproducible code, 19 tests |

## 8. Reproduction

```
py -3 artifacts/dadb_focal_pooling_20261021_0905/scripts/extract_dadb_regional.py
py -3 artifacts/dadb_focal_pooling_20261021_0905/scripts/score_dadb.py
py -3 artifacts/dadb_focal_pooling_20261021_0905/scripts/compute_metrics.py
py -3 -m pytest artifacts/dadb_focal_pooling_20261021_0905/tests -q
```

Total runtime ≈3 minutes on the hardware above. The extraction step is idempotent and
skips if its output exists.

## 9. Safety

No commit, no push, no deletion. No production source, model, checkpoint, raw data, prior
prediction file or existing report was modified — asserted by sha256 in
`tests/test_dadb_focal.py::test_upstream_artifacts_are_unchanged`. Every file written is
inside this run directory. No background jobs remain.
