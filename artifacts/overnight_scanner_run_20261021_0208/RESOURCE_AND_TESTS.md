# Task 6 — Resource and reproducibility audit

Tables: `tables/resource_audit.json`, `test_results.txt`

---

## 1. The memory problem, and how it was solved

Host RAM was **0.33–1.85 GB available of 16.4 GB** for the whole run (88–98% used by other
applications). The TBX11K regional cache is 1.35 GB and the Vietnam cache 555 MB, so the
committed `np.load` path would have swapped heavily or failed.

**5.32 GB of VRAM was free**, so features were streamed from the npz in 256-row blocks
directly to the GPU. Measured cost of loading the 1,356-case REPORT set:

| | |
|---|---|
| file on disk | 0.555 GB |
| wall time | 4.81 s |
| **host RSS increase** | **185 MB** |
| VRAM after load | 0.300 GB |
| VRAM peak during load | 0.488 GB |

Peak VRAM across the whole experiment was **1.48 GB against the 5.15 GB cap** (80% of
6.44 GB). No out-of-memory condition occurred at any point.

Equivalence to the committed path was proved, not assumed: **100.000000% elementwise
identical**, maximum absolute difference **0.0** (`verify_equivalence.py`).

## 2. Matched read-out timing

**Scope, so these are not over-read:** this times the *read-out* over already-cached frozen
features. RAD-DINO inference is identical for both arms and dominates end-to-end latency,
so none of this is a deployment latency figure. Both arms ran in the same process, on the
same 1,356 cases, with 5 warm-up iterations and 20 timed repeats, CUDA synchronised.

| read-out | mean | sd | per case |
|---|---|---|---|
| regional, 144 region scores then max | 9.76 ms | 0.65 ms | **7.2 µs** |
| global, single pooled vector | 0.83 ms | 0.15 ms | **0.61 µs** |

**Regional costs 11.7× the global read-out**, and the cached features are **113× larger**
(555 MB against 4.9 MB for the same 2,708 patients). Both the latency and the storage are
negligible next to the backbone, so this is a storage-and-pipeline consideration, not a
speed objection to focal pooling.

Timing was stable here (sd under 7% of the mean) because warm-up and repeats were
controlled in-process. The *previous* run's test-suite timings varied 196 s → 137 s under
background load and no speedup claim was made from them; none is made here either beyond
the matched in-process comparison above.

## 3. Tests

Full output in `test_results.txt`.

| suite | result |
|---|---|
| new run-directory tests (`tests/test_run_scripts.py`) | **11 passed**, exit 0 |
| streaming-loader equivalence proof | **PASS**, exit 0 |
| full production suite, unmodified | **228 passed, 1 skipped**, exit 0 |
| `py_compile` on every script and test written this run | exit 0 |

The eleven new tests cover the parts of this run that could fail silently:

- npz header parsing matches numpy, and a non-12×12 cache **raises** rather than being
  silently reshaped — the exact failure that cost this project a wasted run before;
- streamed rows are L2-normalised, and the block size does not change which rows come back;
- `agg` matches the committed `agg_np` for every aggregator;
- the projection removes exactly its named subspace and is idempotent;
- TUNE and REPORT do not overlap;
- every TUNE-only artifact declares `report_touched: false`;
- the frozen parameters and manifest **predate** the sealed REPORT pass on disk;
- a per-scanner constant shift provably cannot change within-scanner AUROC — pinned because
  the entire interpretation of the offset arm depends on it.

### One test expectation corrected during the run

Two projection tests first failed with `CUBLAS_STATUS_ALLOC_FAILED`: earlier loader tests
in the same process had left tensors resident and exhausted VRAM. The projection tests are
pure linear algebra, so they were pinned to CPU and an autouse fixture now calls
`torch.cuda.empty_cache()` after each test. This was a defect in the tests, not in the
code under test.

## 4. Confirmation of non-interference

| | |
|---|---|
| repository HEAD at start and end | `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, unchanged |
| branch | `master`, no checkout, reset, stash or rebase |
| `git status --short` at end | only the two untracked run directories |
| commits created | **none** |
| production source, tests, checkpoints, embeddings, predictions, prior reports | **none created, modified or deleted** |
| completed 18-configuration pooling sweep | **not re-run** |
| OOD gate | **not touched**; its advisory default is unchanged and its 11 safety tests pass |
| background processes at end | **none** |

Every file written by this run is inside
`artifacts/overnight_scanner_run_20261021_0208/`.
