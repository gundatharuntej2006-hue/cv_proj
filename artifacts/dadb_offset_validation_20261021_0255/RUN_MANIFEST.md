# Run manifest — DA/DB offset-correction validation, 2026-10-21

**Directory:** `artifacts/dadb_offset_validation_20261021_0255/`
**Repository HEAD, unchanged start to end:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`
(`master`)
**Outcome: NOT EVALUABLE.** Phase 0 precondition 6 failed. **No score was computed on
DA/DB, and no evaluation was run.**

This manifest records the frozen protocol as it stood before the stop, so the decision can
be audited against what was known at the time rather than reconstructed afterwards.

---

## 1. Protocol sources, with hashes

| source | sha256 (first 16) |
|---|---|
| `artifacts/overnight_scanner_run_20261021_0208/NEXT_STEP.md` | `96fb00d728d3cd9d` |
| `artifacts/overnight_scanner_run_20261021_0208/FROZEN_MANIFEST.md` | `1af10d5e64c9762a` |
| `artifacts/overnight_scanner_run_20261021_0208/tables/frozen_parameters.json` | `a632a86c268684ee` |
| `artifacts/overnight_run_20261021_2340/PRIMARY_EXPERIMENT_PREREGISTRATION.md` | `b1b99ba16c199735` |
| `artifacts/p2/external/dadb_third_site.json` | `9badd17376f3ab33` |
| `artifacts/p2/external/radino_dadb_lungs_only_v1.npz` | `b31b10954be72d2a` |
| `scripts/p2_26_dadb_third_site.py` | `9466239df0e783bf` |

Full list with complete digests in `tables/precondition_report.json`. A test asserts every
one of these files still hashes to the recorded value at the end of the run.

## 2. The frozen correction, quoted from source

**Formula**, `FROZEN_MANIFEST.md` §2b:

```
score_corrected = score − offset[rad_system]
```

**Parameters**, `tables/frozen_parameters.json` → `scanner_offsets_tune_non_tb_median`:

| key | offset | estimated on |
|---|---|---|
| `rad_system = 1` | +0.5137025 | Vietnam **TUNE** half, median score of non-TB cases |
| `rad_system = 2` | −0.8610015 | same |
| `rad_system = 3` | +2.3451740 | same |

**The parameters are keyed to Vietnam `rad_system` values.** This matters and is the reason
for the stop; see §5.

**DA/DB-specific estimator**, `NEXT_STEP.md`, quoted verbatim:

> Per site, the median score of that site's non-TB cases. DA/DB has no development half, so
> use leave-one-out: centre each case by the median of the *other* non-TB cases at its site.

## 3. Acceptance rule, quoted from source

`NEXT_STEP.md`, verbatim:

> Accept only if the paired pooled difference is **at least +0.02**, AND its 95% CI
> excludes zero, AND neither site's AUROC falls by more than 0.02.

**Minimum effect size: +0.02 pooled AUROC.** Primary endpoint: pooled AUROC, corrected
minus baseline. Test: paired bootstrap, 4,000 resamples, seed 0, on identical cases.
Guardrail: per-site AUROC must not fall by more than 0.02.

The rule is present and unambiguous. **It was not invented, altered or relaxed by this
run**, and no result was seen before it was read.

## 4. Cohort verification — all checks pass

Reconstructed exactly as `scripts/p2_26_dadb_third_site.py::images()` does: iterate
`sorted(SRC.rglob("*"))` over `data/raw/tbx11k/TBX11K/imgs/extra/da+db`, site from the file
suffix (`.png` → DA, `.jpg` → DB), label from the first character of the filename
(`p` → 1, otherwise 0).

| check | result |
|---|---|
| source directory present | yes |
| **file count** | **176 — reconciles exactly with the reported figure** |
| unique case IDs | 176 of 176, **zero duplicates** |
| missing files | 0 |
| filename prefixes present | only `n` (102) and `p` (74) — no unmapped prefix |
| DA | 72 cases, 22 TB, 50 non-TB |
| DB | 104 cases, 52 TB, 52 non-TB |
| pooled | 176 cases, **74 TB**, 102 non-TB, prevalence 0.4205 |
| against `dadb_third_site.json` | DA (72, 22) ✓ · DB (104, 52) ✓ · DA+DB (176, 74) ✓ |
| patient identifiers | **not distributed**; patient-level overlap is **not checkable** |
| TBX11K subdirectories | 88 under `train/`, 88 under `val/` (TBX11K's own extra-folder structure, not a project split) |

Case-level inventory: `tables/dadb_case_inventory.csv`, 176 rows.

## 5. Provenance of DA/DB — precisely what is and is not known

**Known, and verified here:**

- **Zero overlap with any fitting data.** No DA/DB case ID appears in
  `provisional_c01.csv` (train / development / calibration / internal_test) or in
  `labels_tbx11k_trainval_v1.csv`. The namespaces are disjoint by construction
  (`n1`, `p1`, … against `h0001`, `tb0005`, …).
- **Read by exactly one script.** `git grep -l dadb -- scripts/ src/` returns only
  `scripts/p2_26_dadb_third_site.py`. No selection loop consumes DA/DB.
- **Absent from the decision provenance table.** `artifacts/p2/audit/provenance_table.csv`
  lists fourteen project decisions; DA/DB appears in none of them.
- Committed once, in `75f2e56`, as a third external site.

**Not known, and therefore not claimed:**

- **RAD-DINO checkpoint exposure is UNKNOWN.** The model card states the released
  checkpoint was selected by linear probing on validation sets of *"the evaluation datasets
  described in the paper"*, which the card does not name. DA/DB (NITRD, New Delhi) is not
  among the listed pre-training corpora (MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX), but
  **checkpoint-selection exposure cannot be excluded**.
- **Patient-level overlap with any other cohort cannot be checked**, because no patient
  identifiers ship with DA/DB.

**Therefore DA/DB is described as "never used for any project-specific selection decision",
which is verified, and NOT as "untouched", which is not establishable.** The earlier
`split_provenance.csv` row reading *"not used for any choice"* is accurate at the project
level and should not be read as a claim about the backbone.

## 6. Available artifacts

| artifact | status |
|---|---|
| `radino_dadb_full_v1.npz` | `emb` (176, 768) float32, `n_sanity_fail` 0 — **no `image_id`** |
| `radino_dadb_lungs_only_v1.npz` | `emb` (176, 768) float32, **`n_sanity_fail` 1** — no `image_id` |
| `radino_dadb_border_only_v1.npz` | `emb` (176, 768) float32, `n_sanity_fail` 0 — no `image_id` |
| regional 12×12 DA/DB embedding | **does not exist** |
| DA/DB case-level score file | **does not exist** |

## 7. The stop

Phase 0 precondition 6 failed. The full reasoning is in `BLOCKER_REPORT.md` and the
decision is recorded in `DA_DB_OFFSET_VALIDATION_REPORT.md`. **Nothing was scored.**

## 8. Safety

No commit, no push, no deletion. No production source, checkpoint, historical prediction,
existing report, `NEXT_STEP.md` or preregistration was modified — asserted by hash in
`tests/test_preconditions.py`. Every file written is inside this run directory.
`git status --short` shows only untracked run directories. No background jobs remain.
