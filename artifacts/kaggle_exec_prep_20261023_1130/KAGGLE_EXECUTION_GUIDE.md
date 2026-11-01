# Kaggle execution guide — CLEAR concept audit, Vietnam TUNE only

**Nothing here has been run.** No concept score exists. The reported Vietnam screening
AUROC is **0.7242** and this run cannot change it — it is a diagnostic.

Everything below was verified against the actual files on 2026-10-23. Two upload steps,
one import, six cells.

---

## Before you start — what you are uploading

| | |
|---|---|
| images | 1,352 de-identified chest radiographs (**TUNE only**, 172 MB) |
| clinical rows | 1,352 (**TUNE only**) |
| licence | **CC0-1.0** — Dryad API field for `doi:10.5061/dryad.5tb2rbpcv` |
| **make both datasets Private** | CC0 settles copyright. It does **not** waive ethics, privacy, institutional or Kaggle platform obligations. Nothing goes public |

Vietnam **REPORT** rows, REPORT scores and REPORT images are **not** in the upload. The
staging script filters them out before anything leaves this machine, so the Kaggle session
physically cannot see them. TBX11K `internal_test` is not involved at any point.

---

## Step 1 — Build both uploads (one command)

```bash
cd D:\tb-triage\artifacts\kaggle_exec_prep_20261023_1130

py -3 src/make_upload.py            # dry run: shows what it would build
py -3 src/make_upload.py --write    # builds both folders
```

This produces exactly two folders and nothing else to think about:

```
D:\tb-triage-kaggle-upload\
    code\                           <- the CODE dataset, 10 files, 78 kB
        src\        paths.py  concept_audit.py  preflight.py
                    stage_kaggle_data.py  build_notebook.py  make_upload.py
        configs\    concept_indices.json  model_assets.json
                    tune_cohort_fingerprint.json
        requirements.txt
    data\                           <- the DATA dataset, 1 file
        concept-audit-tune.zip      173 MB
    UPLOAD_CHECKSUMS.txt
```

`code/` contains only what the notebook imports — no `__pycache__`, no tests, no
documentation, and the script aborts if anything else appears in it.

**The data dataset is a single zip.** Kaggle's uploader caps the number of files per
dataset and 1,352 images is over it. PNGs are already compressed, so the archive is
`STORED` rather than deflated: it is a container to get under the file-count cap, not a
size reduction.

A zipped dataset is a **first-class layout**, not a workaround to be undone. The package
reads images straight out of the archive — nothing is extracted, and no second copy of
172 MB of radiographs is written anywhere. If Kaggle auto-extracts it into a folder
instead, that works too; the resolver accepts the archive, a folder, or a folder nested
one level down.

Before it finishes, `make_upload.py` checks the notebook's own discovery globs against
both shapes Kaggle produces, then re-reads the data folder it just wrote. Verified
2026-10-24:

```
notebook discovery contract
  code dataset, contents at dataset root   -> found
  data dataset, contents at dataset root   -> found
  code dataset, wrapped in a folder        -> found
  data dataset, wrapped in a folder        -> found

read-back check
   layout            : staged_zip
   cases             : 1352 (139 TB-positive, 466 previous TB)
   fingerprint       : 1aa55cd5c0eb2088  (matches pinned)
   images            : 0 missing, 0 not approved
```

Dry run verified 2026-10-23:

```
case scores : 2708 rows in source -> 1352 TUNE rows staged
clinical    : 2708 rows in source -> 1352 TUNE rows staged
images      : 1352, 172 MB
fingerprint : 1aa55cd5c0eb2088f2a037d9a8e5770ca1c0f4a75ea21566d743d17577ed1751
fingerprint matches the pinned value: OK
```

It writes `D:\tb-triage-kaggle-staging\concept-audit-tune\`:

```
vietnam_img512_tune/<pid>.png      1,352 files
vietnam_case_scores_tune.csv       1,352 rows
vietnam_meta_tune.csv              1,352 rows
vietnam_tune_report_split.json     the split, ids only, so the guards can still fire
tune_manifest.json                 sorted TUNE ids + sha256
```

The staging directory is **outside** this artifact package on purpose: it holds images and
clinical rows, and the package stays free of both. The script aborts rather than staging a
partial or re-cut cohort.

## Step 2 — Create two private Kaggle datasets

Kaggle → **Datasets → New Dataset** → drag the folder → set **Private** → Create.

| dataset | upload | suggested title |
|---|---|---|
| **data** | the contents of `D:/tb-triage-kaggle-upload/data/` — one zip | e.g. *TB-Triage CLEAR Concept Audit — Vietnam TUNE* |
| **code** | the contents of `D:/tb-triage-kaggle-upload/code/` — `src/`, `configs/`, `requirements.txt` | e.g. *TB-Triage CLEAR Audit Code* |

Upload the code as a dataset rather than pasting it into cells: the notebook imports the
same modules that were tested here, so what runs on Kaggle is what passed the tests.

### Step 2b — Whenever the code in this package changes

The code dataset is a snapshot. If `src/` or `configs/` changes here, that snapshot is
stale and the run fails in a way that looks like a data problem — typically
`No usable input layout found` at cell 3.

Re-run `py -3 src/make_upload.py --write`, then Kaggle → your code dataset →
**New Version** → upload `code\` again. In the notebook, confirm **Add Input** points at
the new version and **restart the kernel**, or the old modules stay imported.

If you are ever unsure which snapshot is up there, compare `UPLOAD_CHECKSUMS.txt`. The
data dataset only needs re-uploading if the cohort itself changes, which would also
change the pinned fingerprint.

## Step 3 — Import the notebook

Kaggle → **Code → New Notebook → File → Import Notebook** → upload

```
artifacts/kaggle_exec_prep_20261023_1130/kaggle/concept_audit_tune_only.ipynb
```

## Step 4 — Settings (all three matter)

| setting | value | why |
|---|---|---|
| **Accelerator** | **GPU** (T4 or P100) | CPU works but is slow |
| **Internet** | **On** | **required.** The pinned CLEAR assets come from huggingface.co and the DINOv2 architecture from github.com at run time. A fully offline run is **not** possible with this package |
| **Persistence** | Variables and Files | lets cell 4 resume instead of restarting |
| **Add Input** | both datasets from Step 2 | the notebook discovers them by content, not by name |

## Step 5 — Run the cells in order

| cell | what it does | what you should see |
|---|---|---|
| **1** | finds the code dataset, imports the modules | `code package : /kaggle/input/...` and the pinned revision |
| **2** | **preflight** — runtime, dependency versions against the pinned ones, resolved model revision and asset checksums, cohort, expected output shape, partition scan | `preflight: GO`. On `STOP` it names the blocker and halts **before** any download |
| **3** | cohort and partition gate | `1352` cases, `139` TB-positive, `466` previous TB, fingerprint `1aa55cd5c0eb2088`, `9` concepts, `cohort gate: PASSED` |
| **4** | **inference.** Fetches ~1.73 GB, encodes 1,352 images at 448 px | a progress line every 256 cases, then `similarities (1352, 9)` |
| **5** | analysis — Spearman + bootstrap CIs + BH, incremental R², distributions, AUROC/AUPRC | four tables printed |
| **6** | validates every expected output, then lists what to download | `output gate: PASSED` |

**Timing is not estimated.** It has never been measured, on this machine or on Kaggle, and
a number invented here would be a guess. Cell 4 prints elapsed minutes as it goes.

If the kernel dies mid-way, just re-run cell 4: it resumes from the last shard. It
**refuses** to resume onto a different id list rather than silently stitching two cohorts.

## Step 6 — Download

From the notebook's **Output** tab:

```
run_manifest.json
tables/concept_associations.csv
tables/concept_incremental_r2.csv
tables/concept_distributions.csv
tables/concept_discrimination.csv
tables/preflight_kaggle.json
```

**Aggregate tables only.** Do not export any per-case file; cell 6 asserts none was
written. Put the downloads in a new timestamped artifact directory here — do not overwrite
this package, and do not edit any earlier run's outputs.

---

## If a gate fires

A raised guard is the package working. None of these should be routed around.

| message | meaning | what to do |
|---|---|---|
| `the code dataset is not attached` | cell 1 found no `*/src/concept_audit.py` | add the code dataset as an input |
| `No usable input layout found` | the data dataset is missing a required file, **or** the code dataset predates zip support | the message lists the folders and zips it searched. If it names your zip, the code dataset is stale — do **Step 2b**. Otherwise rebuild with Step 1 |
| `PREFLIGHT STOP` | a named blocker | fix what it names. Do **not** lower the resolution, swap the architecture or disable a guard |
| `the staged scores file contains N REPORT pids` | the upload was built wrongly | rebuild with `stage_kaggle_data.py`; do not hand-edit the CSV |
| `cohort fingerprint mismatch` | the cohort is not the one this package was prepared for | the split was re-cut. Stop and re-derive the package; a different cohort invalidates the preregistration |
| `WeightLoadError ... RANDOM` | CLEAR's checkpoint keys could not be matched to the encoder | **real blocker, see `PREFLIGHT_REPORT.md` §4.** Report the printed top-level keys. Do not lower `MIN_LOADED_FRACTION` |
| `a partial run exists for a DIFFERENT ordered id list` | resume safety | delete `/kaggle/working/_shards` or re-run cell 3 first |
| `concept embeddings have N rows, expected 368294` | wrong revision | re-pin `142f169c…` |
| CUDA OOM | handled | nothing — the batch halves itself |

## What the run will and will not tell you

**Will:** whether each of 9 concept similarities correlates with the frozen classifier
score among TB-negative participants, whether that differs by previous-TB history, and
whether it adds anything over scanner, age, sex, symptoms and history — with bootstrap CIs
and Benjamini–Hochberg correction.

**Will not:** show that the classifier uses any radiological finding; detect fibrosis;
validate CLEAR; change the screening AUROC. **cavity**, nodule and reticular opacity are
excluded for lacking an exact vocabulary entry and will not be measured at all — cavity
matters clinically in TB, and its absence is a limitation of the audit, not a finding
about the images.

Read `SAFETY_AND_LIMITATIONS.md` before writing up a single number.
