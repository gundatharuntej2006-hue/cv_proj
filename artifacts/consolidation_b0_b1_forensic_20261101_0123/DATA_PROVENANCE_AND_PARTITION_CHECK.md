# Data provenance and partition check

**Git HEAD before:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`
**Working tree before:** 7 tracked files modified by earlier documentation tasks, left
untouched (`tables/git_status_before.txt`).

---

## 1. Every input, hashed

18 inputs, all present, all hashed before use — `tables/provenance_inputs.csv` carries the
full paths, byte counts and sha-256 values. The ones that matter most:

| input | bytes | sha-256 (first 16) | created by |
|---|---|---|---|
| `best_model.pt` | 603,060,057 | `da1617a7d4797066` | HF `peterhan91/CLEAR` @ `142f169c` |
| `concept_embeddings_368294.pt` | 1,131,400,935 | `7241c5b920c4394b` | same |
| `mimic_concepts.csv` | 22,208,546 | `93e975b722422b64` | same |
| B0 scores `vietnam_tune_armA.csv` | 27,451 | `85ef1d2ea0e077a0` | `active_tb_label_raydino_20261024_1254` |
| B1 scores `vietnam_tune_armB1.csv` | 27,814 | `b557baaf8e8a1ed6` | same |
| production `vietnam_case_scores.csv` | 127,937 | `e03c51cf9d2ae9f8` | `overnight_scanner_run_20261021_0208` |
| clinical metadata (Dryad) | 116,309 | `328ee268240f7bdd` | Dryad source |
| TUNE/REPORT allow-list | 27,245 | `0249dcc9c8a8503c` | `artifacts/p2/external` |
| `concept_audit.py` (pinned code) | 25,117 | `ba0ddc3596d8c7ff` | `kaggle_exec_prep_20261023_1130` |
| original 9-concept map | 2,516 | `9b8aad8b017c0331` | same |
| 11-concept map (9 + 2 controls) | 3,817 | `c7efbb10fcacc643` | `b1_clear_mechanism_20261031_2328` |

The first three match `kaggle_exec_prep_20261023_1130/configs/model_assets.json`
byte-for-byte and hash-for-hash, so the model, the embeddings and the vocabulary are
demonstrably the ones the original audit used.

## 2. What was read, and what was not

**Read:** Vietnam **TUNE** only — the 1,352 approved patients on the pinned allow-list,
their images, their Dryad clinical rows, and the B0/B1 per-case scores produced by
`active_tb_label_raydino_20261024_1254`. Plus the CLEAR assets above and the prior runs'
reports and aggregate tables.

**Not read, by assertion and by test:**

| partition | status |
|---|---|
| **Vietnam REPORT** (1,356) | **not read.** The loader selects by the TUNE allow-list; the run asserts `set(pids) & report == {}` and halts otherwise. Verified: **0 REPORT pids in cohort** |
| **TBX11K `internal_test`** (901) | **not read, not referenced.** No TBX11K artifact is among the 18 inputs |

A note worth making explicitly: the repo image directory holds all **2,708** Vietnam
images, so REPORT images sit on disk beside the approved ones. The run reports `1,352
approved, 0 missing, 1,356 present but unapproved` — the unapproved 1,356 are enumerated
only to prove they were *not* opened, and `concept_audit.open_image` is called solely for
ids on the allow-list.

## 3. Cohort

| | |
|---|---|
| cohort fingerprint | `1aa55cd5c0eb2088f2a037d9a8e5770ca1c0f4a75ea21566d743d17577ed1751` — **matches** the pinned `tune_cohort_fingerprint.json` |
| definition | sha-256 of the newline-joined, sorted TUNE pid strings |
| n | **1,352** patients |
| Xpert-positive | **139** (prevalence 0.1028) |
| TB-negative | **1,213** |
| previous TB | **466** |
| duplicates / missing / non-finite | **none** — `load_tune_only` raises on each |
| images missing | **0** |

Labels (`xpert_result`) and `past_tb` are re-derived from the Dryad clinical CSV by the
pinned loader, not copied from a prior run's table. The B0 and B1 arm files' own label
columns are checked against that derivation and must agree exactly, or the run halts.

**Alignment is by patient id, never by row position.** Arm scores are re-indexed onto the
loader's ordered id list through a dictionary; a missing or duplicated id is a blocker.
The ordered-id hash equals the cohort fingerprint here because the score file is already
sorted by pid.

## 4. B0 is production-identical — evidence, not assertion

| quantity | value |
|---|---|
| Spearman, B0 vs production `score_baseline` | **1.000000** |
| Pearson | **1.000000** |
| maximum absolute difference | **1e-06** |

1e-06 is the six-decimal rounding of the stored CSV. Arm A is a deterministic
reproduction of the production fitting path (same frozen RAD-DINO features, same regional
12x12 max head, same five seeds), so **B0 is the production model, not an approximation
of it**, and B1 is the experimental active-TB-label refit. Sources are named in
`tables/metric_reproduction.json` under `b0_identity_with_production`.

## 5. Concept family — the 9 / 11 / 17 reconciliation

| run | encoded | roles |
|---|---|---|
| `concept_audit_result_20261024_0130` (original Kaggle audit) | **9** | 9 primary. cavity, nodule, reticular opacity excluded — no exact vocabulary entry, deliberately not approximated |
| `b1_clear_mechanism_20261031_2328` | **11** | the same 9 primary **plus 2 negative controls** (`scoliosis`, `degenerative changes`), declared before that run |
| **this run** | **17** | the same 11 **plus 6 diagnostic probes** |

**Nothing was dropped or redefined.** The nine primary indices are byte-identical across
all three maps, and a test asserts it. The "8 of 11" figure in the previous report counts
the 11 encoded concepts of that run (9 primary + 2 controls), of which 8 shifted with
intervals excluding zero — 7 primary and 1 control.

**The Benjamini-Hochberg family remains the 9 primary concepts**, as predeclared, in every
run. Controls and probes are outside it and carry no adjusted p-value. The family is
**not independent** — see `tables/concept_similarity_matrix.csv` — so the correction is
conservative for a correlated family rather than a correction for nine independent tests.

### The six diagnostic probes, and when they were fixed

They are listed in `configs/concept_indices.json` with a written declaration recorded
**before any image-side similarity for them was computed**: two negation pairs
(`no consolidation`, `no scarring`) and four normality anchors (`clear lungs`,
`lungs are clear`, `no acute cardiopulmonary process`, `normal chest`).

They were selected from the **text-side** geometry — a property of the vocabulary and the
embedding tensor alone, involving no patient data and no classifier score — to test the
hypothesis the task brief names in §3 A4. They are **diagnostic only**: excluded from the
BH family, not endpoints, and barred from informing any intervention, threshold, subgroup
or model choice.

`no scarring` is included deliberately so the negation test is applied to the project's
headline concept and not only where a null result would be convenient. `normal chest` is
the normality anchor whose text-side cosine to the consolidation anchor is **lowest**
(+0.100), so the probe set is not stacked toward agreement.

## 6. Privacy and export

| | |
|---|---|
| per-case concept similarities | computed into `D:\tb-triage-scratch\consolidation_forensic_20261101_0123`, **outside** this run directory, and **deleted** when the run ends |
| per-case output in this directory | **none.** A test asserts no file here contains a TUNE pid, no `.npy` exists, and no table has a row count that could be per-case |
| images, clinical rows, embeddings | **not copied, not exported** |
| uploads | **none.** Inference ran locally; nothing was sent to Kaggle or any external service |
| external contact | **none** |

## 7. Modification guard

No training, no fine-tuning, no head fitting, no change to production inference, weights,
preprocessing, thresholds, pooling or calibration. No commit, push or deletion. The
audited package `kaggle_exec_prep_20261023_1130` is **imported, never edited** — the
extended concept map is supplied by shadowing its asset lookup at run time with a
temporary file that is removed afterwards.

Before/after hashes and `git status` are recorded in `tables/`.
