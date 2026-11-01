# Phase 0 — Repository state and evidence audit

**Run start:** 2026-10-21T10:23:51+05:30 · **Run directory:**
`artifacts/realworld_generalization_20261021_1015/`

---

## 1. Git state at start

| | |
|---|---|
| HEAD | `f878a6c9e9746803938463941ae1ffaa5d1fd96a` (`master`) |
| modified, tracked | `docs/p2_external_validation.md`, `docs/p2_vietnam_result.md`, `report/p2_00_abstract_intro.md`, `report/p2_99_conclusion.md`, `report/paper_draft.md`, `report/slides/p2_slides_outline.md` |
| untracked run dirs | `overnight_run_20261021_2340`, `overnight_scanner_run_20261021_0208`, `dadb_offset_validation_20261021_0255`, `dadb_focal_pooling_20261021_0905`, `doc_finalisation_20261021_0930` |

**The six modified files are pre-existing user-visible changes from the preceding
documentation task. They were preserved untouched by this run**, as were all five prior run
directories. Nothing was committed, pushed, reverted or deleted.

## 2. Resources

| | at start | mid-run |
|---|---|---|
| host RAM | 4.18 GB available of 16.39 (74% used) | 4.32 GB |
| VRAM | 5.32 GB free of 6.44 (80% cap 5.15) | peak used **0.94 GB** |
| disk D: | 188 GB free | — |
| GPU / torch | RTX 4050 Laptop · torch 2.6.0+cu124, CUDA available | |

RAM was healthier than in earlier runs (4.2 GB against 0.33 GB), but the streaming loader
was used regardless since it is proved equivalent and costs nothing.

## 3. Verified against the prior evidence summary

| prior claim | verification |
|---|---|
| Vietnam screening AUROC ≈ 0.7242 | **confirmed** — `mil_final.json`, Vietnam REPORT, n=1,356, 141 TB, grid 12 / max / seeds 17–79. Baseline arm in this run reproduces the full-cohort figure to 0.0005 (0.7245 vs 0.7240) |
| TBX11K ≈ 0.9949 | confirmed as the **full-image, all-negatives** figure; under the current configuration TBX11K development is 0.9368 (regional) / 0.9570 (global) |
| pooling deltas Shenzhen +0.0279, Montgomery +0.0401, DA −0.0009, DB +0.0717 | **confirmed** — `cross_dataset_metrics.csv`, `dadb_focal_pooling_…/tables/metrics.json` |
| negatives 0.7791 → 0.9172 on DA/DB | **confirmed**, and the 0.7791 side reproduces exactly; it is a training-composition effect, seed-17 on both sides |
| scanner probe 0.9737 balanced accuracy on TUNE | **confirmed** — `overnight_run_…/source_probe_metrics.json`; chance 0.3381 |
| scanner thresholds cut disparity but scanner 3 sensitivity 0.674 → 0.488 | **confirmed** — `threshold_report_metrics.csv` |
| subspace removal rejected, Δ ≈ −0.0002 | **confirmed** — `acceptance_verdict.json`, CI [−0.0009, +0.0004] |
| scanner offset +0.0206, CI crossing zero, p = 0.1575 | **confirmed** — `robustness_checks.json` |
| OOD gate advisory by default | **confirmed** — `src/p2_cls/pipeline.py:150`; 11 safety tests pass; **not touched by this run** |
| TBX11K has no scanner/site field | **confirmed** — ten columns, none of them a source field |
| DA/DB regional embeddings now exist | **confirmed** — extracted in the preceding run to that run's directory, not to `artifacts/p2/regional/` |
| the documentation task changed reporting only | **confirmed** — `git diff` touches six `.md` files and nothing else |

**One correction to the brief's framing.** It describes Vietnam REPORT as if it retained
read-once status. It does not: REPORT has now been scored in at least four separate runs.
This is recorded in the exposure registry and treated accordingly throughout.

## 4. Exposure registry

`tables/cohort_exposure_registry.csv`, 12 rows. Summary of what may still be used for what:

| partition | still usable as an unexposed test? |
|---|---|
| **TBX11K internal_test (901)** | **YES — the project's only unexposed partition. Deliberately not read in this run.** |
| TBX11K train / development / calibration | no — training, early stopping, thresholds |
| Vietnam TUNE | no — selected the pooling grid and aggregator, scanner offsets, thresholds |
| Vietnam REPORT | no — scored repeatedly across four runs |
| Shenzhen, Montgomery | no — informed preprocessing and negative composition |
| DA, DB | informed no configuration decision, but now scored once; **single-use status spent** |
| RAD-DINO checkpoint | **UNKNOWN** — the model card does not name its checkpoint-selection datasets, so exposure to any public TB benchmark cannot be excluded |

## 5. Tests at start

Production suite **228 passed, 1 skipped**. OOD non-suppression **11 passed**. No
production code was read into this run's path and none was modified.

## 6. Reconciliation note

Per the brief, no tracked report was edited in this task. The one inconsistency worth
recording — that the preceding run's preregistration stated 93 `latent_tb` training rows
where the true count is **94** — is documented in
`03_EXISTING_DATA_EXPERIMENT_REPORT.md` §5 and corrected only inside this run directory.
