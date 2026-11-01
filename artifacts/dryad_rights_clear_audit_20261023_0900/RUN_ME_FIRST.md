# Run me first

The TUNE-only CLEAR concept audit. **It has not been executed.** The data licence and the
concept mapping are resolved; execution is blocked only on host memory on this machine.

---

## Which route

| route | when | what you need |
|---|---|---|
| **Local** | once this machine has ~4 GB of free RAM | nothing to upload |
| **Kaggle** | any time | two private datasets, below |

Measured at preparation time: **0.77–0.84 GB RAM available of 16.39, 95% used**. CLEAR
needs a 603 MB checkpoint plus a slice of a 1,131 MB tensor, so it was not run here.

## Local

```bash
cd D:\tb-triage

# dry validation: no download, no inference, no images
py -3 artifacts/dryad_rights_clear_audit_20261023_0900/src/run_concept_audit.py

# synthetic tests (20, ~9 s)
py -3 -m pytest artifacts/dryad_rights_clear_audit_20261023_0900/tests -q

# the real run - downloads ~1.73 GB on first use, then encodes 1,352 images
py -3 artifacts/dryad_rights_clear_audit_20261023_0900/src/run_concept_audit.py --run-inference
```

**Runtime is not estimated**, because it has not been measured. VRAM headroom was 5.3 GB
and the batch halves automatically on OOM.

## Kaggle

1. **Code dataset (private):** `src/`, `configs/`, `requirements.txt` from this directory.
2. **Data dataset (private):** the files below. The Vietnam images are **CC0-1.0**, so
   this is permitted — see `DATA_RIGHTS_ASSESSMENT.md`. **Set the dataset Private
   anyway**: CC0 settles copyright, not patient sensitivity.

| required path | from | rows / files |
|---|---|---|
| `vietnam_img512/<pid>.png` | `data/processed/v1/vietnam_img512` | 2,708 |
| `vietnam_case_scores.csv` | `artifacts/overnight_scanner_run_20261021_0208/tables/` | 2,708 |
| `Lunit_Versions_-_Dryad_Data_Set.csv` | `data/raw/vietnam/` | 2,708 |
| `vietnam_tune_report_split.json` | `artifacts/p2/external/` | TUNE 1,352 / REPORT 1,356 |

**Do not attach anything from TBX11K `internal_test`.** Cell 2 hard-fails if an attached
input exposes it.

3. Notebook settings: **GPU** on, **internet on** (CLEAR downloads from Hugging Face),
   persistence **Variables and files**.
4. Open `kaggle/concept_audit_tune_only.ipynb` and run cells in order. Each gate prints
   PASSED or raises.
5. Download `tables/concept_associations.csv` and `tables/concept_incremental_r2.csv`.
   **These are aggregate. Do not export any per-case table.**

## What the run will and will not tell you

**Will:** whether each of 9 concept similarities correlates with the frozen classifier
score among TB-negative participants, whether that differs by previous-TB history, and
whether it adds anything over scanner, age, sex, symptoms and history — with bootstrap
CIs and BH correction.

**Will not:** show that the classifier uses any radiological finding; detect fibrosis;
validate CLEAR; change the screening AUROC, which stays **0.7242**. Three concepts —
**cavity**, nodule, reticular opacity — are excluded for lacking an exact vocabulary
entry and will not be measured at all.

## If something fails

| symptom | cause |
|---|---|
| `SplitViolation` | REPORT contamination or a missing/duplicate TUNE patient. Working as intended |
| `attached input exposes the locked partition` | remove `internal_test` from your dataset |
| `concept embeddings have N rows, expected 368294` | wrong revision; re-pin `142f169c…` |
| CUDA OOM | handled; batch halves automatically |
| swapping / very slow | not enough host RAM. Use Kaggle |
