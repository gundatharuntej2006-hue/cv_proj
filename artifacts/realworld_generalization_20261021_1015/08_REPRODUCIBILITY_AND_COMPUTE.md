# Phase 5 — Reproducibility and compute audit

Machine-readable: `tables/compute_audit.json`, `tables/production_suite.txt`.

---

## 1. Environment

Python 3.13.2 · torch 2.6.0+cu124 · numpy 2.2.6 · scikit-learn 1.9.0 · scipy 1.17.1 ·
pandas 3.0.3. GPU: NVIDIA GeForce RTX 4050 Laptop, 6.44 GB.
Windows-11-10.0.26200-SP0.

## 2. Resources used

| | |
|---|---|
| host RAM available | 4.18 GB at start, 4.32 GB mid-run, of 16.39 GB |
| **peak VRAM allocated** | **0.94 GB** against the 5.15 GB cap (80% of 6.44) — within cap |
| streaming load, 2,708 cases | 4.60 s, **host RSS increase 94 MB** for a 555 MB file |
| Phase 2 total runtime | 440 s for four arms × five seeds, including all bootstraps |

Four arms × five seeds × up to 120 epochs on cached features cost **seven minutes** on a
laptop GPU. Worth stating plainly: **compute has never been the binding constraint on this
project.** Data is.

## 3. Cached-feature equivalence

The streaming GPU loader was re-verified against the committed
`p2_32_mil_sweep.repool(X, 12, 12)` on a fixed sample during this run:

| | |
|---|---|
| elementwise identical | **100.000000%** |
| max absolute difference | **0.0** |
| max difference in the max-pooled score | 7.15e-07 |

**End-to-end check.** Arm H — the baseline label definition — reproduces the published
Vietnam full-cohort AUROC to **0.0005** (0.7245 against 0.7240). Earlier runs matched to
four decimals, so this is run-to-run GPU nondeterminism in head training rather than a
pipeline difference. It is smaller than every effect discussed, and is flagged rather than
hidden. The script refuses to proceed if the gap exceeds 0.01.

## 4. Read-out latency, matched conditions

2,708 cases, 5 warm-up iterations, 20 timed repeats, CUDA synchronised, in one process:
**17.73 ms mean (sd 0.24), 6.55 µs per case.**

Scope: read-out over cached frozen features only. **RAD-DINO inference is excluded and
dominates end-to-end latency**, so this is not a deployment figure and no speedup is
claimed from it.

## 5. Tests

| suite | result |
|---|---|
| this run (`tests/test_phase2.py`) | **12 passed** |
| production suite, unmodified | **228 passed, 1 skipped** |
| `dadb_focal_pooling` run-directory tests | 19 passed |
| `overnight_scanner_run` run-directory tests | 11 passed |
| `py_compile` on this run's scripts | clean |

**Suites were run separately.** Running two run-directory suites in one pytest process
causes CUDA allocation contention on a 6.4 GB GPU — a harness issue documented in the
preceding run, not a code defect.

This run's tests assert, among other things, that the preregistration is byte-identical to
what was hashed before scoring; that no Vietnam label entered head fitting; that
`internal_test` was not read; that no two arms share a score array; that the reported
metrics recompute from the saved per-case predictions; that the decision rule was applied
as written and did not return ACCEPT; that arm X is arithmetically explained by arms L and
S; and that `past_tb` is a variable documented in the Dryad README rather than inferred
from a filename.

**One test was corrected during the run.** It first required all 2,708 scores in every arm
to be unique; arm X has one exact tie at the six-decimal precision the CSV stores. That is
a rounding artefact, not array reuse — which a separate test checks directly — so the
assertion now allows up to five ties. The defect was in the test, not in the result.

## 6. Processes

No background job was started by this run and none remains. **The two long-running
`verdant-ai` uvicorn processes predate this session by several days and were deliberately
left untouched**, as were all other unrelated processes.

## 7. What was and was not modified

| | |
|---|---|
| production source, checkpoints, caches, predictions, prior reports | **unmodified** |
| the six tracked docs modified by the preceding task | **preserved exactly as they were** |
| five prior run directories | **untouched** |
| commits, pushes, deletions | **none** |
| files written | all inside `artifacts/realworld_generalization_20261021_1015/` |
