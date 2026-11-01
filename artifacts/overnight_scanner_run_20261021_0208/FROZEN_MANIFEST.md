# Frozen experiment manifest — written before any REPORT metric was computed

**Frozen at:** 2026-10-21, after the TUNE-only phase and before `task_report_sealed.py` ran.
**Repository HEAD:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, branch `master`, clean apart
from two untracked run directories.

Everything below — every parameter, threshold, offset, seed, metric and acceptance
criterion — was fixed from TUNE data alone. REPORT was scored exactly once after this file
was written, and nothing in it was changed afterwards.

**Why one sealed REPORT pass covers all three arms.** The runbook asks separately for a
frozen plan before Task 2's REPORT scoring and before Task 3's. Reading REPORT twice is
strictly worse than reading it once, so all arms are frozen here and scored in a single
pass. No arm's REPORT result can influence another's design, because all designs predate
this file.

---

## 1. Provenance of the score

All scores come from the 12×12 regional max-pooling head, trained on **TBX11K train only**
(3,241 rows, 541 TB, healthy-only negatives), early-stopped on TBX11K development (659 rows,
59 TB), ensembled over seeds **17, 29, 43, 61, 79**. No Vietnam TB label enters head
training.

**Pipeline equivalence was verified before any new number was trusted:**

| check | expected | obtained |
|---|---|---|
| streaming GPU loader vs committed `repool(X,12,12)` | identical | **100.000000% elementwise identical**, max abs difference 0.0 |
| TBX11K development AUROC | 0.9368 | **0.9368** |
| Vietnam TUNE AUROC, unprojected | 0.7237 | **0.7237** |

A bounded-memory implementation was required because host RAM was 0.33–1.84 GB of 16.4 GB
while the TBX11K regional cache is 1.35 GB. Features stream from the npz in 256-row blocks
straight to the GPU; host memory never holds more than ~56 MB. Peak VRAM 1.48 GB against a
5.15 GB cap (80% of 6.44 GB).

## 2. Frozen parameters

Exact values in `tables/frozen_parameters.json`.

### 2a. Thresholds — Task 2

Procedure: the empirical 0.70 quantile (`numpy.quantile`, linear interpolation between
order statistics) of the scores of **TUNE** cases with `xpert_result == 0`, taken within the
relevant group. Target specificity 70% is the project's existing convention and is labelled
**exploratory**.

| arm | threshold |
|---|---|
| **A — single pooled threshold** | 2.4766000 |
| **B — scanner-specific**, rad_system 1 | 2.0070364 |
| **B — scanner-specific**, rad_system 2 | 1.3887347 |
| **B — scanner-specific**, rad_system 3 | 3.7074950 |

Sampling uncertainty, stratified bootstrap on TUNE non-TB scores, 2,000 resamples, seed 17
(`tables/threshold_bootstrap.csv`):

| group | n non-TB in TUNE | 95% CI | width | flag |
|---|---|---|---|---|
| rad_system 1 | 182 | [1.5761, 2.4424] | 0.866 | **small subgroup** |
| rad_system 2 | 590 | [1.0093, 1.7890] | 0.780 | |
| rad_system 3 | 441 | [3.3539, 4.0130] | 0.659 | |
| pooled | 1,213 | [2.2723, 2.6903] | 0.418 | |

Every scanner-specific threshold is estimated from fewer cases than the pooled one, so each
is correspondingly less stable. The rad_system 1 interval is the widest and its subgroup the
smallest.

### 2b. Scanner-offset correction — Task 5

A per-scanner additive shift, **not** a threshold change. Unlike a threshold, this can
change AUROC, because it re-orders cases across scanners.

`score_corrected = score − offset[rad_system]`, offsets being the **TUNE** non-TB median:

| scanner | offset |
|---|---|
| 1 | +0.5137025 |
| 2 | −0.8610015 |
| 3 | +2.3451740 |

Operating threshold for the corrected score, taken the same way on TUNE: **1.7206912**.

### 2c. Scanner-subspace removal — Task 3, preregistered

Implements `artifacts/overnight_run_20261021_2340/PRIMARY_EXPERIMENT_PREREGISTRATION.md` §4.

| parameter | value |
|---|---|
| transform | project out the scanner subspace from each 768-d token, before pooling |
| subspace fitted on | **Vietnam TUNE tokens only**, `rad_system` labels only |
| TB labels used to fit the transform | **none** |
| rank | **3**, as preregistered |
| estimator | 3-class linear layer on token vectors, AdamW, lr 1e-2, wd 1e-4, 60 epochs, batch 64 images, seed 17 |
| basis | right-singular vectors of the class-mean-centred weight matrix, orthonormality residual 4.77e-07 |
| head seeds | 17, 29, 43, 61, 79 |
| aggregation | max over 144 regions |
| basis hash | `84c7a540805b4c91` (`tables/scanner_subspace_basis.npy`) |

**Known defect in the preregistered rank, recorded before seeing REPORT.** A 3-class
centred weight matrix has rank 2, and the third singular value came out exactly **0.0**.
The preregistered rank of 3 therefore removes two genuine scanner directions plus one
arbitrary direction from the null space. The preregistration said rank 3, so **rank 3 is
what is scored and judged**; the defect is reported, not corrected after the fact.

**Arms:**

- **P0 baseline** — no projection. Reference arm.
- **P1 preregistered** — head trained on unprojected TBX11K, projection applied to Vietnam
  tokens at inference. This is the arm §4 specifies ("the head is not retrained") and the
  only one judged against the acceptance criteria.
- **P2 secondary, exploratory** — head retrained on projected TBX11K tokens. Not covered by
  the preregistration; reported for completeness only and cannot be declared a success.

### 2d. Acceptance criteria — fixed, not to be reinterpreted

From §4 verbatim: *accept only if the **worst-scanner** AUROC improves **and** the paired
pooled difference excludes zero.*

Operationalised, on REPORT:

1. **Criterion 1** — worst-scanner AUROC under P1 > worst-scanner AUROC under P0.
2. **Criterion 2** — the 95% paired-bootstrap CI for AUROC(P1) − AUROC(P0), pooled, 4,000
   resamples, seed 0, excludes zero.
3. Both must hold. A pooled gain with worst-scanner degradation is a **rejection**.
4. The measured per-scanner noise floor on this cohort is ±0.055 AUROC between random
   halves, so any per-scanner movement below that is reported as noise regardless of sign.

## 3. What REPORT will be used for, and what it will not

**Will:** compute, once, for each frozen arm — pooled and per-scanner AUROC and AUPRC with
bootstrap CIs; paired AUROC differences against P0; sensitivity, specificity, referral rate,
PPV and NPV at the thresholds frozen in §2a and §2b.

**Will not:** select among arms, adjust any threshold or offset, re-fit any calibrator, or
re-define any acceptance criterion.

**Standing limitation.** Vietnam TUNE chose the pooling configuration in earlier work, so
REPORT sits inside a cohort that already influenced the design. Every REPORT number here is
a **controlled comparison, not an independent estimate of generalisation**, and is labelled
exploratory.

## 4. Input hashes

| file | sha256 (first 16) |
|---|---|
| `scripts/common.py` | `d1a654e9c7734dc2` |
| `scripts/task1_2_tune_diagnostics.py` | `1495304892a0fd9b` |
| `scripts/task3_subspace_tune.py` | `f799c20beb982ed4` |
| `scripts/task4_representation_diagnostics.py` | `69456122c1a7c199` |
| `scripts/task4b_inlp.py` | `8a4d8ffca4153025` |
| `scripts/task5_calibration.py` | `ce9d47faaa83e69e` |
| `scripts/verify_equivalence.py` | `c88f1b4f35181477` |
| `tables/vietnam_case_scores.csv` | `e03c51cf9d2ae9f8` |
| `tables/scanner_subspace_basis.npy` | `84c7a540805b4c91` |

Upstream inputs are hashed in `preflight.json`.
