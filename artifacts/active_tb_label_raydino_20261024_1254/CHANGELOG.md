# Change log

Everything created by this run, in the order it was created. **Nothing outside this
directory was written, edited or deleted.**

| # | path | what it is |
|---|---|---|
| 1 | `tables/git_status_before.txt`, `tables/git_head_before.txt` | repository state captured before any work |
| 2 | `tables/protected_hashes_before.txt` | sha-256 of 14 protected artifacts, before |
| 3 | `scripts/phase01_audit.py` | Phase 0/1 inventory, read-only |
| 4 | `tables/partition_table.csv`, `target_crosswalk.csv`, `near_duplicate_cross_split.csv`, `phase01_audit.json` | its outputs |
| 5 | `PHASE0_STATE_AND_PROVENANCE.md` | Phase 0 report |
| 6 | `LABEL_SEMANTICS.md` | Phase 1 report and decision gate |
| 7 | `PROTOCOL.md` | preregistration |
| 8 | `tables/protocol_hash.txt` | its sha-256 + timestamp, **written before any fit** |
| 9 | `scripts/embed_calibration.py` | extends the frozen feature cache to the 899 calibration images |
| 10 | `outputs/regional_tbx11k_calibration_lungs_g12_v1.npz` | those features, (899, 144, 768) fp16 |
| 11 | `outputs/embed_calibration.json` | the bit-for-bit verification against 24 cached development images |
| 12 | `scripts/phase2_label_alignment.py` | the experiment |
| 13 | `tables/metrics_by_arm.csv`, `paired_differences.csv`, `per_seed_results.csv`, `score_distribution_by_class.csv`, `posthoc_class_separation.csv`, `run_manifest.json` | its outputs |
| 14 | `outputs/predictions/calibration_arm{A,B1,B2}.csv`, `vietnam_tune_arm{A,B1,B2}.csv` | per-case, per-seed predictions |
| 15 | `ACTIVE_TB_LABEL_ALIGNMENT_REPORT.md` | Phase 2 report |
| 16 | `tests/test_run.py` | 19 tests |
| 17 | `PHASE3_CLEAR_LINKAGE.md` | Phase 3 — missing dependency, aggregates only |
| 18 | `PHASE4_RAYDINO_ASSESSMENT.md` | Phase 4 — blockers, existing package re-verified |
| 19 | `REPRODUCE.md` | exact commands and environment |
| 20 | `FINAL_REPORT.md` | the end-of-run answers |
| 21 | `tables/protected_hashes_after.txt`, `git_status_after.txt`, `git_head_after.txt` | after-state |
| 22 | `CHANGELOG.md`, `FILE_MANIFEST.sha256` | this file and the manifest |

## Corrections made during the run

- **140, not 139.** A test caught that the C01 mismatch set is 140 images, not 139: 139
  carry only obsolete-TB boxes and C01 adds `tb1199`. `PHASE0` §1 and `LABEL_SEMANTICS`
  §3 were corrected. `PROTOCOL.md` was **not** touched — its §1 says "139 of 800 TBX11K
  TB images carry only ObsoletePulmonaryTuberculosis boxes", which is accurate as
  written, and editing a hashed protocol after fitting is exactly what the hash exists to
  prevent.
- **The first attempt at the primary endpoint aborted**, because the frozen regional
  feature cache turned out to cover only train + development. Resolved by extending the
  cache inside this run directory, with a reproduction check, rather than by falling back
  to the early-stopping-contaminated development split.
- **The experiment was run twice**, the second time only to add per-case prediction files
  and the post-hoc class-separation table. Every metric was byte-identical.

## Protected state

| | before | after |
|---|---|---|
| git HEAD | `f878a6c9e9746803938463941ae1ffaa5d1fd96a` | identical |
| `git status --short` | 7 modified tracked files from earlier tasks | **identical** |
| 14 protected artifact hashes | — | **identical** (`diff` clean) |

No commit, push or deletion. No process terminated. No network access beyond the local
Hugging Face cache at `D:\hf-cache`, which already held RAD-DINO.
