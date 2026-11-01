# Result integrity check — cross-dataset pooling comparison

**Owner:** Person 2 · **Date:** 2026-10-21 · Read-only verification. No predictions,
checkpoints, caches or prior artifacts were modified.

## 1. Comparator identity — a labelling error in the earlier report, now corrected

The earlier report labelled arm A **"global (deployed)"**. That label is **wrong** and is
superseded here.

**Verified from source:** `scripts/p2_36_cross_dataset_pooling.py` fits arm A with
`H.train_seeds(Gtr, ytr, Gdv, ydv, ...)` on `radino_tbx11k_lungs_only_v1.npz` — i.e. it is
**newly refitted** under the matched configuration (lungs-only masking, healthy-only
negatives, seeds 17/29/43/61/79, early stop on TBX11K development).

**It is not the historical deployed checkpoint.** `artifacts/p2/bundle/head.npz`
(sha256 `14a7e6cf38a2eff2`) carries `variant: full_clahe0` — the older full-image,
all-negatives configuration — and was deliberately **not used**, because doing so would
confound pooling with masking and negative composition. That checkpoint is preserved
untouched and was not loaded by this experiment.

**Correct name: "matched global-pooling baseline."** All three arms are refitted identically
and differ only in representation/aggregation, which is what makes the comparison matched.
No claim is made that arm A reproduces the historically deployed system's behaviour.

## 2. The identical 0.9690 — explained, not a defect

The two cohorts' regional AUROCs are **not identical**. They differ in the fifth decimal and
coincide only after rounding to four:

| cohort | regional max AUROC (6 dp) | as reported (4 dp) |
|---|---|---|
| Shenzhen | **0.969024** | 0.9690 |
| Montgomery | **0.968966** | 0.9690 |

A coincidence of rounding on 662 and 138 cases respectively. The earlier report should have
shown more precision; this document supersedes it.

## 3. Independent metric recomputation

Recomputed from `cross_dataset_case_predictions.csv` using **`sklearn.metrics.roc_auc_score`**,
an implementation independent of the project's `selection.auc`:

| cohort | arm | project impl | sklearn | agree |
|---|---|---|---|---|
| Shenzhen | A matched global | 0.941097 | 0.941097 | yes |
| Shenzhen | B matched mean | 0.847055 | 0.847055 | yes |
| Shenzhen | C regional max | 0.969024 | 0.969024 | yes |
| Montgomery | A matched global | 0.928879 | 0.928879 | yes |
| Montgomery | B matched mean | 0.877155 | 0.877155 | yes |
| Montgomery | C regional max | 0.968966 | 0.968966 | yes |

**Agreement to six decimal places on all six figures.** The reported values reproduce.

## 4. Alignment, mapping and leakage checks

| check | result |
|---|---|
| rows / unique `image_id` | 800 / 800 — no duplicates |
| stored label == label independently derived from the filename suffix | **True for all 800** |
| cohort == filename prefix (`shz`→Shenzhen, `mcu`→Montgomery) | **True for all 800** |
| Shenzhen composition | 336 TB / 662 — matches the NLM published counts |
| Montgomery composition | 58 TB / 138 — matches the NLM published counts |
| arm score arrays pairwise identical? | **A≠B, A≠C, B≠C** — no array reuse |
| cohort score arrays | distinct, 662 and 138 values respectively |
| alignment method | join by stable `image_id`, not row or directory order |

**Positive-class mapping** was re-derived from the filename suffix rather than trusted from
the stored column, and matches in every case. Patient identity: the NLM sets are one image
per patient, so `image_id` is the patient key and no within-cohort patient overlap is
possible by construction. Cross-cohort overlap is excluded by disjoint filename prefixes.

Training-set overlap was verified separately and earlier: maximum cosine similarity between
any NLM image and any TBX11K training embedding is 0.8389, zero pairs above 0.90.

## 5. Paired confidence intervals, recomputed

Matched per-case predictions, 4,000 bootstrap resamples, seed 0, both classes required in
each resample:

| cohort | comparison | ΔAUROC | 95% CI |
|---|---|---|---|
| Shenzhen | C − A (matched global) | +0.0279 | [+0.0128, +0.0434] |
| Shenzhen | C − B (matched mean) | +0.1220 | [+0.0938, +0.1501] |
| Montgomery | C − A (matched global) | +0.0401 | [+0.0089, +0.0767] |
| Montgomery | C − B (matched mean) | +0.0918 | [+0.0375, +0.1503] |

**Reproduces the reported intervals to four decimal places.**

## 6. Artifact hashes

| artifact | sha256 (first 16) |
|---|---|
| `cross_dataset_manifest.json` | `610af8f0330af697` |
| `cross_dataset_case_predictions.csv` | `0687a7dcf38e736a` |
| `cross_dataset_metrics.csv` | `1ecbcf80282c5885` |
| `regional_nlm_lungs_g12_v1.npz` | `62df11dbdd984c37` |
| `regional_tbx11k_lungs_g12_v1.npz` | `3da5665a14433c10` |
| `scripts/p2_36_cross_dataset_pooling.py` | `067754599bbb25bd` |
| `artifacts/p2/bundle/head.npz` (NOT used) | `14a7e6cf38a2eff2` |

## 7. Verdict

The numerical results **reproduce independently**. One reporting defect was found and is
corrected here: **arm A is a matched refitted global-pooling baseline, not the deployed
checkpoint**, and the earlier "global (deployed)" label is withdrawn.

The identical-looking 0.9690 values are a rounding artifact of two genuinely different
numbers, not evidence of array reuse, cohort mix-up or leakage.

**Commands used:**
```
py -3 -c "<recomputation shown in this document>"   # sklearn cross-check, alignment, paired CIs
```
All checks are read-only and rerunnable from the committed prediction file.
