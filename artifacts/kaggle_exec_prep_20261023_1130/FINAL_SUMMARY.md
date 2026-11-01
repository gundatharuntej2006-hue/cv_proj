# Final summary — Kaggle execution preparation

**Run:** `artifacts/kaggle_exec_prep_20261023_1130/` · 2026-10-23
**HEAD before and after:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` — unchanged

> **No concept inference was executed. No concept score exists. No metric changed.**
> The reported Vietnam screening AUROC remains **0.7242**.

---

## Completed

| | |
|---|---|
| **Package and interface verified** | Read the previous package end to end. The named docs `README_KAGGLE.md`, `KAGGLE_INPUT_CHECKLIST.md`, `DATA_RIGHTS_AND_MODEL_LICENCES.md`, `RUN_PROTOCOL.md` and `DECISION_RULES.md` belong to the **earlier** package `kaggle_concept_raydino_20261021_1830`, not to the CLEAR one; both were read. The CLEAR package's own entry point is `RUN_ME_FIRST.md`, and its executable flag is `--run-inference` on `src/run_concept_audit.py` |
| **Pinned revision resolved** | `HfApi.repo_info` returns sha `142f169c8b006b0f28cf55c6a237c3003db5ed8f`; three asset checksums captured and now compared by preflight. The `mimic_concepts.csv` hash matches the vocabulary the 9 concept indices were mapped against |
| **Licences** | Vietnam images CC0-1.0 (dataset-specific Dryad API field); CLEAR Apache-2.0 for code and weights; RayDINO untouched |
| **Internet requirement established** | **Required.** Weights from huggingface.co, DINOv2 architecture via `torch.hub` from github.com. A fully offline run is **not** possible with this package, and the preflight says so in a field a test asserts |
| **Previous tests re-run** | **20 passed**, unchanged, nothing edited in that package |
| **Three concrete defects found and fixed** | below |
| **Zipped uploads supported** | added 2026-10-24, after the 1,352-image folder exceeded Kaggle's per-dataset file cap. `staged_zip` is a first-class layout: images are read from the archive, nothing is extracted. Verified against the real 173 MB upload — 1,352 members, 0 unapproved, fingerprint matched |
| **New tests** | **36 passed** in 13 s |
| **Staging verified by dry run** | 2,708 → 1,352 TUNE rows in both CSVs, 1,352 images, 172 MB, fingerprint matches the pinned value |

### The four defects

1. **The previous module could not have run on Kaggle at all.** It derived every input
   path from `HERE.parents[2]`, which resolves to `/kaggle` when `src/` sits in an input
   dataset. Cell 2 would have failed on the first data access. `paths.py` now resolves the
   layout and raises an actionable `LayoutError` instead.
2. **A wrong checkpoint key layout would have been silent.** The old loader filtered by
   name and used `strict=False`; an unmatched layout yields an empty dict, loads nothing,
   and encodes with a **randomly initialised** ViT. `_load_encoder` now matches on name
   **and shape** across six key roots × six prefixes, reports the fraction matched, and
   **raises below 90 %**.
3. **A zipped upload shared its scratch directory by filename.** The extracted CSVs were
   cached under the archive's stem, so two different `upload.zip` files would have shared
   a cohort — the second run reading the first one's data and passing every guard. Found
   by a test, fixed by keying the scratch directory to the archive's path, size and mtime.
4. **A test in the previous package asserted a guard that could never fire.** Carried
   forward honestly: the REPORT-intrusion check sat behind an allow-list, so it was dead
   code. Here it is live in the staged layout — where a REPORT row genuinely means a
   wrongly built upload — and documented as inert in the repository layout rather than
   presented as protection it does not provide.

## Prepared, not run

| deliverable | state |
|---|---|
| `src/stage_kaggle_data.py` | dry-run verified; **you run it with `--write`** |
| `src/preflight.py` | run locally, verdict **STOP on RAM alone**, checksums verified |
| `src/concept_audit.py` | dry validation passes: 1,352 cases, 139 positive, 466 previous TB, 9 of 12 concepts, fingerprint matched, 0 missing images |
| `kaggle/concept_audit_tune_only.ipynb` | 6 code cells, every one syntax-checked; a test asserts the file on disk is what its generator produces |
| `KAGGLE_EXECUTION_GUIDE.md` | literal steps, settings, gate-by-gate failure table |
| `EXPECTED_OUTPUTS.md` | 6 files, exact columns, exact row counts and group sizes, validation rules |
| `PREFLIGHT_REPORT.md` | verified prerequisites, the RAM blocker, the residual checkpoint-key risk |
| `SAFETY_AND_LIMITATIONS.md` | partition controls with an honest can-it-fire column, licensing caution, 10 interpretation boundaries |
| `RUN_MANIFEST_TEMPLATE.json` | the shape `run_manifest.json` must take |
| `MANIFEST.sha256` | code and config checksums. **No image, clinical row, weight or patient identifier is in this package** — enforced by test |

## Not run

- **CLEAR concept inference.** Blocked locally on measured host RAM: **0.58–1.48 GB
  available of 16.39, 94–96 % used**, against ~3 GB needed. Not blocked on rights, not on
  mapping, not on GPU.
- Everything excluded by instruction: Vietnam REPORT, TBX11K `internal_test`, RayDINO,
  feature subtraction, apical weighting, subgroup Platt scaling, retraining, threshold
  optimisation.

## What changed on disk

| | |
|---|---|
| model predictions, checkpoints, caches | **unchanged** |
| tracked source and documents | **unchanged** — `git status` shows only the new untracked directory added to the 7 doc edits already pending from earlier tasks |
| previous artifact directories | **unchanged**; `dryad_rights_clear_audit_20261023_0900/MANIFEST.sha256` re-verified, 0 discrepancies |
| commits, pushes, uploads, emails | **none** |
| written outside the package | **nothing.** The staging directory does not exist until you run `--write` |

## The one risk that survives into the run

CLEAR's `best_model.pt` key layout is undocumented and could not be verified without
downloading 603 MB into 0.7 GB of free RAM. If the search in `_load_encoder` does not
reach 90 %, **cell 4 stops** and prints the checkpoint's actual top-level keys. That is the
one place this package may still need a code change — and the threshold should be extended
to fit the keys, never lowered to fit the threshold.

## Single next action

Run **Step 1** of `KAGGLE_EXECUTION_GUIDE.md`:

```bash
py -3 artifacts/kaggle_exec_prep_20261023_1130/src/stage_kaggle_data.py --write
```

then upload the two private datasets and run the notebook. Expect a diagnostic, not an
improvement — and read `SAFETY_AND_LIMITATIONS.md` §5, which states the likely result in
advance so that a confirmation is not mistaken for a discovery.
