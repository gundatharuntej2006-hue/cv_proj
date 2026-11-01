# Kaggle package: concept audit + RayDINO vs RAD-DINO

**This package was prepared locally. It has NOT been run on Kaggle. No Kaggle GPU run has
occurred and no Kaggle result exists.** Everything below is setup instruction.

---

## What this is

Two staged, isolated experiments, both frozen-encoder only:

1. **Concept audit (diagnostic).** Whether high-scoring TB-negative cases — especially
   those with documented previous TB — associate with radiological concepts such as
   fibrosis, volume loss, calcification, cavity and opacity, using CLEAR. **A concept
   audit is not expected to change AUROC.**
2. **Controlled benchmark.** RayDINO against RAD-DINO, frozen, with a matched downstream
   head. **RayDINO may improve, match or worsen performance. Nothing here assumes it is
   better.**

Stage 3, a single targeted intervention, is **disabled by default** and must not be run
until Stages 1 and 2 justify it in writing.

## Before you start: three hard blockers

| blocker | status | what to do |
|---|---|---|
| **Vietnam image licence** | **BLOCKED.** The Dryad record names no specific licence | Resolve with the depositors before uploading any image anywhere. Do not assume CC0 |
| **RayDINO weights** | **GATED, and redistribution unresolved** | Request access at `ai.meta.com/resources/models-and-libraries/raydino-downloads/`. Weights are under the FAIR Noncommercial Research License. Read it before attaching them to a Kaggle dataset |
| **CLEAR commit** | no revision published on the model card | Pin a specific commit of `github.com/peterhan91/CLEAR` and record it in `manifest.json` |

Full detail: `DATA_RIGHTS_AND_MODEL_LICENCES.md`. **Unknown means blocked.**

## Kaggle notebook settings

| setting | value | why |
|---|---|---|
| Accelerator | **GPU T4 x2** or **P100** | notebook 02 encodes thousands of images |
| Internet | **On** for RAD-DINO and CLEAR weights; **Off** is fine if you attach all weights | RAD-DINO and CLEAR download from Hugging Face |
| Persistence | **Variables and files** | so interrupted feature extraction resumes |
| Environment | pin to the versions in `requirements.txt` if results must be reproduced | |

## Input dataset structure

Attach the code as one dataset and your data as another, private, dataset. The notebooks
discover inputs by searching `/kaggle/input/**`, so the exact dataset names do not matter —
but the **relative paths inside them do**:

```
/kaggle/input/<your-code-dataset>/
    src/{guards,encoders,evaluate}.py
    configs/{target_label,benchmark}.yaml
    manifest.json                      <- you fill this from manifest_template.json
/kaggle/input/<your-private-data>/
    vietnam_img512/<pid>.png           (2,708 files)  -- only if the licence permits
    vietnam_case_scores.csv            (2,708 rows)
    Lunit_Versions_-_Dryad_Data_Set.csv
    vietnam_tune_report_split.json
    provisional_c01.csv
    labels_tbx11k_trainval_v1.csv
```

**Never attach anything from the `internal_test` partition.** Notebook 00 fails if an
attached dataset exposes it.

Exact file list, checksums and shapes: `KAGGLE_INPUT_CHECKLIST.md`.

## Run order

1. **`kaggle/00_preflight.ipynb`** — validates runtime, dependencies, paths, licence
   flags, splits, labels, IDs and shapes. **If anything fails, stop here.**
2. **`kaggle/01_concept_audit.ipynb`** — runs the score-only subgroup analysis on Vietnam
   **TUNE** regardless, and runs concept inference **only** if the manifest clears the
   image licence and images are attached. Otherwise it reports the blocker and computes
   nothing. It will not fabricate concept scores.
3. **`kaggle/02_raydino_vs_raddino.ipynb`** — extracts frozen features per encoder under
   each encoder's own official preprocessing, fits the matched head on TBX11K **train**,
   selects regularisation on **development**, and scores. The RayDINO arm is **skipped,
   not faked**, if `RAYDINO_WEIGHTS` is unset.
4. **`kaggle/03_optional_targeted_ablation.ipynb`** — **disabled**. Leave it so.

## Outputs

Everything is written to `/kaggle/working/`:

```
preflight.json
tables/concept_score_summary.csv            (only if unblocked)
tables/concept_model_score_associations.csv (only if unblocked)
tables/subgroup_metrics.csv
tables/encoder_comparison.csv
features/*.npz + *.json                     (per encoder, resumable)
heads/*.joblib                              (fitted head weights, not just seeds)
```

Download them from the notebook's Output tab. **Do not** download anything derived from
patient images unless you have settled the licence question.

## Partition discipline, enforced in code

| partition | use |
|---|---|
| TBX11K **train** | head fitting |
| TBX11K **development** | regularisation choice, thresholds, calibration — **nothing else** |
| TBX11K **calibration** | available, unused by default |
| TBX11K **`internal_test`** | **LOCKED. Never attached, enumerated, loaded or scored.** `guards.assert_path_allowed` raises on any path containing it |
| Vietnam **TUNE** | exploratory diagnostics only; already used repeatedly in prior work, so **not independent validation** |
| Vietnam **REPORT** | **not read by any notebook here.** Not for selection, tuning, thresholds or evaluation |

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `attach the code dataset` | the notebook found no `*/src/guards.py` under `/kaggle/input`. Attach the code dataset |
| `LockedPartitionError` | something referenced `internal_test`. This is working as intended — remove it from your attached data |
| `target_label.yaml missing - refusing to train` | attach `configs/`. Notebook 02 will not train on an undefined target |
| `RayDINO weights not supplied` | expected. Request gated access and set `RAYDINO_WEIGHTS` |
| `set the RayDINO input resolution from its model card` | the resolution is deliberately not guessed. Read it and set it |
| CUDA OOM during extraction | handled automatically — batch size halves and semantics are unchanged. Also enable persistence so progress resumes |
| extraction restarts from zero | enable **Variables and files** persistence; the checkpoint lives beside the output `.npz` |
| notebook 01 says BLOCKED | the manifest does not clear the image licence. That is the designed behaviour, not a bug |

## What this package deliberately will not do

Download gated weights for you · upload anything to Kaggle · read API credentials ·
touch `internal_test` · read Vietnam REPORT · fabricate a concept score · claim a Kaggle
run happened · recommend deployment.
