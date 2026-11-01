# Run manifest — overnight scanner run, 2026-10-21

**Directory:** `artifacts/overnight_scanner_run_20261021_0208/`
**Start:** 2026-10-21T02:08:12+05:30 · **End:** 2026-10-21T02:46 · **Elapsed ≈38 min**
**HEAD at start and end:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, branch `master`
**Working tree at start:** clean apart from `artifacts/overnight_run_20261021_2340/`

Machine-readable preflight, including sha256 for all 10 upstream inputs: `preflight.json`.
Frozen experimental parameters and acceptance criteria: `FROZEN_MANIFEST.md`.

## Resources

| | at start | note |
|---|---|---|
| host RAM available | **0.33 GB** of 16.39 GB (98% used) | rose to 1.85 GB mid-run; never above 2 GB |
| VRAM free | 5.32 GB of 6.44 GB | 80% cap = 5.15 GB; peak used **1.48 GB** |
| disk D: free | 188 GB | run directory is ~1 MB |
| GPU | RTX 4050 Laptop | CUDA available |
| Python | 3.13.2, torch 2.6.0+cu124, scikit-learn 1.9.0 | |

The RAM figure is why this run exists in the form it does. The previous run recorded the
subspace experiment as blocked for want of about 2 GB. It was unblocked by streaming
features from the npz in 256-row blocks directly to the GPU, proved equivalent to the
committed path before use.

## Split roles, verified rather than assumed

| partition | n | role | fitted on here? |
|---|---|---|---|
| TBX11K train | 3,241 (541 TB) | head training | yes |
| TBX11K development | 659 (59 TB) | early stopping | yes |
| Vietnam TUNE | 1,352 (139 TB) | all transforms, thresholds, offsets | yes |
| Vietnam REPORT | 1,356 (141 TB) | scored once, after the manifest was frozen | **no** |

TUNE and REPORT do not overlap; asserted in code and in `tests/test_run_scripts.py`.

## Order of operations, which is the point

1. Preflight and input hashes, before anything else.
2. Streaming-loader equivalence proof — **100.000000% elementwise identical**.
3. TUNE-only phase: diagnostics, thresholds, subspace fit, INLP sweep, calibration.
4. `FROZEN_MANIFEST.md` written, with every parameter and both acceptance criteria.
5. **One** sealed REPORT pass covering all arms. One read, not three.
6. Robustness checks, tests, reporting.

A test asserts that `frozen_parameters.json` and `FROZEN_MANIFEST.md` predate
`acceptance_verdict.json` on disk.

## Deliverables

| file | contents |
|---|---|
| `RUN_MANIFEST.md` | this file |
| `preflight.json` | HEAD, environment, resources, split roles, 10 input hashes |
| `FROZEN_MANIFEST.md` | frozen parameters, thresholds, offsets, acceptance criteria |
| `THRESHOLD_RESULTS.md` | Task 2 |
| `SUBSPACE_RESULTS.md` | Task 3, verdict REJECT |
| `REPRESENTATION_DIAGNOSTICS.md` | Task 4, including the corrected INLP analysis |
| `CALIBRATION_AND_ERROR_ANALYSIS.md` | Task 5 |
| `RESOURCE_AND_TESTS.md` | Task 6 |
| `FINAL_REPORT.md`, `NEXT_STEP.md` | Task 7 |
| `test_results.txt` | all four suites, captured |
| `tables/` | result csv / json / npy files |
| `scripts/` | 9 python files, all syntax-checked |
| `tests/` | 11 tests, all passing |

## Self-corrections made during the run, recorded rather than hidden

1. **The rank sweep in Task 4 was degenerate.** Ranks 8/16/32/64 returned byte-identical
   results, because a 3-class weight matrix yields only three singular vectors. Replaced
   with iterative nullspace projection in `task4b_inlp.py`; the degenerate table is kept
   with the correction attached.
2. **Two new tests failed with `CUBLAS_STATUS_ALLOC_FAILED`**, because earlier loader tests
   left tensors resident on the GPU. The projection tests were pinned to CPU and an autouse
   fixture now frees VRAM. A defect in the tests, not in the code under test.
3. **The preregistration's rank of 3 is wrong for three classes** — the third singular
   value came out exactly 0.0. Scored as preregistered regardless; the defect is reported
   in `SUBSPACE_RESULTS.md` §2 rather than patched after seeing results.

## Confirmation

No commits, no pushes, no deletions, no production file touched, no background process left
running. `git status --short` shows only the two untracked run directories.
