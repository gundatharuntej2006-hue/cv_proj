# Reproduce

Total wall time on the machine below: **about 5 minutes**, plus 23 s for the RayDINO
package tests. No Kaggle, no network, no gated weights.

## Environment

| | |
|---|---|
| OS | Windows 11, `D:\tb-triage` |
| Python | 3.13.2 (CPython, MSC v.1942, 64-bit) |
| torch | 2.6.0+cu124, CUDA **available** |
| numpy | 2.2.6 |
| pandas | 3.0.3 |
| GPU | CUDA device present; the embedding step uses fp16 on GPU, the head trains on GPU |
| `HF_HOME` | `D:\hf-cache` (shared model cache; RAD-DINO is already resident) |

CPU-only works: `p2_32_mil_sweep.train` falls back to CPU, and
`p2_31_regional_mil.RegionalEmbedder` falls back to fp32. Expect roughly 10x the time for
the embedding step. The fp32 fallback will **not** reproduce the fp16 cache bit-for-bit,
and `embed_calibration.py` will refuse to proceed — that refusal is the intended
behaviour, not a bug.

## Inputs this run reads (none are written)

| path | role |
|---|---|
| `artifacts/p2/dryrun/provisional_c01.csv` | split and category definitions |
| `artifacts/p2/labels_tbx11k_trainval_v1.csv` | cross-checked, not used for targets |
| `artifacts/p2/dedup/provisional_near_dup_official_v0.csv` | leakage inventory |
| `artifacts/p2/regional/regional_tbx11k_lungs_g12_v1.npz` | frozen features, train + development |
| `artifacts/p2/regional/regional_vietnam_lungs_g12_v1.npz` | frozen features, Vietnam |
| `artifacts/p2/external/vietnam_tune_report_split.json` | the TUNE / REPORT split |
| `data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv` | Xpert outcomes |
| `data/processed/v1/tbx11k/img512/`, `data/processed/v1/masks/` | calibration PNGs + lung masks |
| `data/raw/tbx11k/TBX11K/annotations/json/TBX11K_{train,val}.json` | the COCO category declarations |

Pre-run hashes of the protected artifacts: `tables/protected_hashes_before.txt`.

## Commands, in order

```sh
cd D:/tb-triage

# 1. Phase 0 / 1 — partition inventory, target crosswalk, leakage inventory.  ~2 s
py -3 artifacts/active_tb_label_raydino_20261024_1254/scripts/phase01_audit.py

# 2. Extend the frozen feature cache to the 899 calibration images.  ~1.1 min, GPU
#    Verifies itself against 24 cached development images first and aborts on mismatch.
py -3 artifacts/active_tb_label_raydino_20261024_1254/scripts/embed_calibration.py

# 3. The preregistered experiment: 3 arms x 5 seeds + bootstraps.  ~2.7 min
#    Refuses to start if PROTOCOL.md no longer matches tables/protocol_hash.txt.
py -3 artifacts/active_tb_label_raydino_20261024_1254/scripts/phase2_label_alignment.py

# 4. Tests for this run: target mapping, split guards, checkpoint/shape assertions,
#    metric consistency.  19 tests, ~2 s
py -3 -m pytest artifacts/active_tb_label_raydino_20261024_1254/tests -q

# 5. The pre-existing RayDINO package, re-verified today.  29 tests, ~23 s
py -3 -m pytest artifacts/kaggle_concept_raydino_20261021_1830/tests -q
cd artifacts/kaggle_concept_raydino_20261021_1830 && \
  tr -d '\r' < MANIFEST.sha256 | sha256sum -c -        # 28/29; FINAL_RUN_REPORT.md
                                                       # post-dates the manifest
```

## Expected output of step 3

```
Vietnam TUNE: 1352 participants, 139 Xpert-positive (REPORT untouched)
arm A    baseline, tb_label                            n_train 3241  pos 541
arm B1   active_tb_label, latent kept as negative      n_train 3241  pos 447
arm B2   active_tb_label, latent excluded              n_train 3147  pos 447

arm  calAUROC    AUPRC     sens     spec
A      0.9498   0.6881   0.8519   0.9156
B1     0.9460   0.6689   0.8642   0.9071
B2     0.9493   0.6832   0.8519   0.9169

PRIMARY  B1 - A  on calibration vs active_tb_label:
  delta -0.0037  95% CI [-0.0081, +0.0002]  half-width 0.0041
  VERDICT: REJECT
```

Determinism: every seed is pinned (heads 17/29/43/61/79; bootstrap 0) and the run was
executed twice during development with byte-identical metrics. `torch.manual_seed` plus
fixed data order is sufficient here because the head is a single linear layer with no
dropout and no data augmentation. Figures may move in the last decimal on a different
GPU or CUDA version.

## Guards that will stop a wrong run

| guard | where | fires when |
|---|---|---|
| protocol hash | `phase2_label_alignment.check_protocol` | `PROTOCOL.md` edited after hashing |
| `internal_test` size | `phase2`, `embed_calibration`, `tests` | the split file no longer holds 901 rows |
| `internal_test` absent from every loaded embedding | `phase2`, `tests` | any locked image is in a feature file |
| Vietnam REPORT absent | `phase2`, `tests` | a REPORT pid enters the scored set |
| split sizes 5941 / 659 / 899 | `phase2` | the split file changed |
| cache reproduction | `embed_calibration` | re-embedding does not match the frozen cache |
| shape / normalisation | `embed_calibration`, `tests` | not (899, 144, 768), or regions not L2-normalised |
| threshold provenance | `tests` | calibration specificity lands exactly on 0.90, which would mean the threshold was fitted there |
| AUROC consistency | `tests` | the reported AUROC cannot be recomputed from the preserved per-case predictions |

## What this run does not touch

No tracked file is modified. No commit, push or deletion. No existing script, checkpoint,
embedding, prediction, manifest or prior artifact directory is written to.
`artifacts/p2/regional/` is read only — the calibration features land in
`outputs/regional_tbx11k_calibration_lungs_g12_v1.npz` inside this run directory.

**TBX11K `internal_test` and Vietnam REPORT were not read.**
