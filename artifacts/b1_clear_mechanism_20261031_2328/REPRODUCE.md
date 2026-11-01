# Reproduce

One command. **35 minutes** on the machine below (1.9 min encoding, 33 min bootstraps),
plus a one-off 1.7 GB model download. No Kaggle, no upload, no gated weights.

## Environment

| | |
|---|---|
| OS | Windows 11 (10.0.26300), repo at `D:\tb-triage` |
| Python | 3.13.2 |
| torch | 2.6.0+cu124, CUDA 12.4 |
| GPU | NVIDIA GeForce RTX 4050 Laptop |
| numpy / scipy / scikit-learn | 2.2.6 / 1.17.1 / 1.9.0 |
| `HF_HOME` | `D:\hf-cache` (shared cache; never delete) |

## One-off setup

```sh
# The CLEAR package, pinned by commit. --no-deps matters: CLEAR's own pins would
# otherwise reinstall torch.
py -3 -m pip install --no-deps \
  "git+https://github.com/peterhan91/CLEAR.git@57d2eae57ddabf8e56c8013ec675336d71b3d8e7"
py -3 -m pip install ftfy regex
```

The run downloads `best_model.pt` (603 MB) and `concept_embeddings_368294.pt` (1.13 GB)
from `peterhan91/CLEAR` at revision `142f169c8b006b0f28cf55c6a237c3003db5ed8f` and hashes
both before use.

**If the download stalls at 0 bytes**, the Xet transfer backend is the cause. Force the
plain HTTPS path:

```sh
HF_HUB_DISABLE_XET=1 py -3 -c "
import os; os.environ['HF_HOME']=r'D:\hf-cache'
from huggingface_hub import hf_hub_download
hf_hub_download('peterhan91/CLEAR','concept_embeddings_368294.pt',
                revision='142f169c8b006b0f28cf55c6a237c3003db5ed8f')"
```

`mimic_concepts.csv` is expected at `D:\hf-cache\clear_vocab\mimic_concepts.csv`
(sha-256 `93e975b7…`); fetch it from the same repo and revision if absent.

## Run

```sh
cd D:/tb-triage/artifacts/b1_clear_mechanism_20261031_2328/scripts

# Validate every prerequisite without touching the GPU. ~20 s.
py -3 run_mechanism.py

# Full run: encode, gate, three arms, paired bootstrap. ~35 min.
py -3 run_mechanism.py --run-inference

# Tests. ~30 s.
cd D:/tb-triage && py -3 -m pytest artifacts/b1_clear_mechanism_20261031_2328/tests -q
```

Options: `--device cpu` (far slower; will not reproduce the fp16 figures),
`--batch N` (halves itself on OOM), `--keep-scratch` (keeps the per-case shards —
**do not use for anything that leaves this machine**).

## Expected output

```
vocabulary verified: 93e975b722422b64, 4+2 pinned indices re-derived
TUNE cohort        : 1352 patients, 139 TB-positive, 466 with previous TB
cohort fingerprint : 1aa55cd5c0eb2088 (matches pinned)
concepts           : 11 (9 primary + 2 negative controls: scoliosis, degenerative changes)
images             : 1352 approved, 0 missing, 1356 present but not approved
REPORT pids loaded : 0  |  internal_test referenced: no
locked thresholds  : B0 -0.5973, B1 -2.1911  (TBX11K development, never TUNE)
asset verified     : best_model.pt                 da1617a7d4797066
asset verified     : concept_embeddings_368294.pt  7241c5b920c4394b
  loaded via clear.load_pretrained; visual tower has 87.2M parameters
  gate scarring_rho      published 0.5661  local 0.5661  diff 0.000008  PASS
  ... (6 of 6)
  reproduction gate PASSED
  paired bootstrap: 10000 replicates, 0 degenerate resamples discarded

concept                  rho B0   rho B1     delta 95% CI
scarring                 0.5661   0.5317   -0.0344 [-0.0428, -0.0265]
fibrosis                 0.4166   0.3984   -0.0182 [-0.0269, -0.0103]
scoliosis                0.0184   0.0114   -0.0070 [-0.0153, +0.0013]  <- NEGATIVE CONTROL
degenerative changes    -0.1381  -0.1764   -0.0384 [-0.0475, -0.0297]  <- NEGATIVE CONTROL
```

Determinism: every seed is pinned (per-arm 20261023, paired 20261031, both 10,000
replicates). Figures may move in the last decimal on a different GPU or CUDA version —
the reproduction gate exists precisely to decide whether such drift matters, and its
tolerances are declared in `ANALYSIS_MANIFEST.md` §7.

## Guards that will stop a wrong run

| guard | fires when |
|---|---|
| analysis-manifest hash | `ANALYSIS_MANIFEST.md` edited after preregistration |
| vocabulary sha-256 + index re-derivation | the concept vocabulary is not the pinned one |
| asset sha-256 and byte count | checkpoint or embeddings truncated or substituted |
| cohort fingerprint | the analysis cohort is not the approved TUNE set |
| duplicate / missing / non-finite | any broken row in scores or clinical metadata |
| REPORT pid in cohort | a Vietnam REPORT participant enters the run |
| arm label agreement | an arm's labels disagree with the audited cohort |
| encoder fail-closed | no `encode_image`, or a visual tower without a projection |
| similarity shape | not (1352, 11), or non-finite |
| **reproduction gate** | the local pipeline does not reproduce the published figures |
| resume-hash mismatch | a partial run exists for a different ordered id list |

## What this run does not touch

No tracked file, checkpoint, prediction file, feature cache, threshold or prior artifact
directory is written. The audited package `kaggle_exec_prep_20261023_1130` is **imported,
never edited**. Per-case concept similarities are computed into
`D:\tb-triage-scratch\b1_clear_mechanism_20261031_2328` and **deleted** when the run ends;
a test asserts the scratch is gone and that no written file contains a patient identifier.

**Vietnam REPORT and TBX11K `internal_test` were not read.**
