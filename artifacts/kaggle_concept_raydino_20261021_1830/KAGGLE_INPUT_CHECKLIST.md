# Kaggle input checklist

Exact files to attach, their required **relative paths inside the dataset**, and expected
shapes. Machine-readable with live checksums: `tables/kaggle_input_checklist.csv`.

**The repository already holds every data file below.** What it does not hold is
permission to redistribute some of them — see the Status column.

---

## A. Code dataset — attach from this package, nothing to source

| relative path | contents |
|---|---|
| `src/guards.py` | locked-partition guards, shape and ID assertions |
| `src/encoders.py` | frozen-encoder loaders, resumable extraction, declared pooling |
| `src/evaluate.py` | matched head, metrics, paired bootstrap, threshold discipline |
| `configs/target_label.yaml` | the target mapping, identical for both arms |
| `configs/benchmark.yaml` | seeds, grid, pooling rule, bootstrap settings |
| `manifest.json` | **you fill this** from `manifest_template.json` |

## B. Private data dataset — you attach, after settling permissions

| required relative path | kind | size | expected shape / rows | status |
|---|---|---|---|---|
| `vietnam_case_scores.csv` | file | 0.13 MB | **2,708 rows**: `pid, xpert_result, rad_system, split_half, score_baseline` | repo has it |
| `vietnam_tune_report_split.json` | file | 0.03 MB | `tune` 1,352 · `report` 1,356 · overlap 0 | repo has it |
| `provisional_c01.csv` | file | 0.57 MB | usable splits only: train 5,941 · development 659 · calibration 899 | repo has it — **strip the 901 `internal_test` rows before uploading** |
| `labels_tbx11k_trainval_v1.csv` | file | 0.54 MB | **8,400 rows** | repo has it |
| `Lunit_Versions_-_Dryad_Data_Set.csv` | file | 0.12 MB | **2,708 rows**, 14 columns including `past_tb` | repo has it — **licence BLOCKED** |
| `vietnam_img512/<pid>.png` | dir | 346 MB | **2,708 files** | repo has it — **licence BLOCKED** |
| `vietnam_img512_lungs/<pid>.png` | dir | 149 MB | 2,708 files | repo has it — **licence BLOCKED**, optional |
| TBX11K images (`health`, `sick`, `tb`) | dirs | ~2.9 GB | 3,800 / 3,800 / 800 | repo has it — **terms unverified, so blocked** |

## C. Model weights

| component | how to get it | attach or download? |
|---|---|---|
| **RAD-DINO** | `microsoft/rad-dino`, revision `110cbc18d5133582e320b43d53bf5c44e410c936` | **download in-notebook** with internet on. MIT |
| **CLEAR** | `peterhan91/CLEAR` + a pinned commit of `github.com/peterhan91/CLEAR` | **download in-notebook**. Apache-2.0. Pin the commit yourself |
| **RayDINO** | request form at `ai.meta.com/resources/models-and-libraries/raydino-downloads/` | **GATED.** Set `RAYDINO_WEIGHTS` to the local path, or pass the temporary emailed URL. **Read the FAIR Noncommercial Research License before re-hosting it as a Kaggle dataset — that permission is not established** |

## D. Before you upload anything derived from patient images

- [ ] Vietnam image licence resolved with the depositors — the Dryad record **names no
      licence**, so this is currently **BLOCKED**
- [ ] TBX11K redistribution terms checked
- [ ] your institution's rules on third-party platforms checked
- [ ] the Kaggle dataset is set **Private**
- [ ] the `internal_test` rows are stripped from `provisional_c01.csv`
- [ ] `manifest.json` filled honestly — **`unknown` blocks the concept stage by design**

If any box is unticked, attach only the non-image files. The score-only subgroup analysis
in notebook 01 still runs; concept inference does not, and will not be faked.

## E. What the notebooks verify on your behalf

Notebook 00 checks: GPU and internet, package versions, that no attached dataset exposes
`internal_test`, the manifest licence flags, usable split counts, the target mapping,
image counts, and synthetic shape assertions. **If it fails, stop.**
