# Final run report

**Run path:** `D:\tb-triage\artifacts\kaggle_concept_raydino_20261021_1830\`
**Date:** 2026-10-21, 19:22 → 20:15 +05:30
**Git HEAD before and after:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` — **unchanged**

## This run prepared a package. It did not run on Kaggle.

**No Kaggle notebook was executed. No Kaggle GPU was used. No Kaggle output exists.** No
concept score and no encoder benchmark number was produced anywhere. Saying otherwise
would be false.

---

## 1. Git status

| | |
|---|---|
| HEAD before / after | identical |
| tracked files modified by this run | **none** |
| `git status --short` before / after | **byte-identical.** The run directory appears as `?? artifacts/kaggle_concept_raydino_20261021_1830/` in both, because the baseline was captured immediately after the directory was created and git reports an untracked directory as one line regardless of its contents. The only change inside the working tree is the contents of that one untracked directory |
| commits, pushes, deletions | **none** |
| background jobs left running | **none** |

Seven tracked files appear modified by **earlier** documentation tasks and were left
untouched.

## 2. Files created — 27, all inside the new run directory

| group | contents |
|---|---|
| reports | `PRE_FLIGHT_AUDIT.md`, `CONCEPT_AUDIT_REPORT.md`, `RAYDINO_COMPARISON_PROTOCOL.md`, `FINAL_RUN_REPORT.md` |
| governance | `DATA_RIGHTS_AND_MODEL_LICENCES.md`, `RUN_PROTOCOL.md`, `DECISION_RULES.md` |
| Kaggle | `README_KAGGLE.md`, `KAGGLE_INPUT_CHECKLIST.md`, `LOCAL_DRY_RUN.md`, `manifest_template.json`, `requirements.txt` |
| notebooks | `kaggle/00_preflight.ipynb`, `01_concept_audit.ipynb`, `02_raydino_vs_raddino.ipynb`, `03_optional_targeted_ablation.ipynb` (disabled) |
| code | `src/guards.py`, `src/encoders.py`, `src/evaluate.py`, `src/stage0_audit.py`, `src/build_notebooks.py` |
| config | `configs/target_label.yaml`, `configs/benchmark.yaml` |
| tests | `tests/test_package.py` — **29 tests** |
| tables | input inventory, preflight audit, input checklist, git state |
| integrity | `MANIFEST.sha256` |

**No production file, checkpoint, embedding, prediction or prior report was modified.**

## 3. What was actually tested, and what was not run

**Run and verified locally:**

- Stage 0 audit — input inventory with checksums, split re-derivation, target-label
  semantics, encoder comparability, licence verification from official sources.
- Stage 1 **score-only** subgroup analysis on Vietnam **TUNE** — committed scores and
  metadata only; no model, no image.
- **29 synthetic tests**, all passing: locked-partition guards (5), encoder shape
  assertions for both dimensions and grids, `pool_to_grid` output shape, normalisation and
  wrong-grid rejection, ID-alignment detection of reordering, duplicates and length
  mismatch, target-mapping rejection of undefined and overlapping classes, threshold
  refusal outside `development`, threshold accuracy, metric correctness, empty and
  single-class groups, paired bootstrap on a real difference and a null, **patient-level
  resampling widening the interval versus case-level**, BH correction, C-selection on
  development only, notebook JSON validity, and a check that the package contains no
  image, weight or `.npz`.
- `py_compile` clean on every module.

**Not run, and why:**

| not run | reason |
|---|---|
| CLEAR concept inference | **Vietnam image licence unresolved** — the Dryad record names no licence. Unknown blocks |
| RayDINO feature extraction | **weights gated**; redistribution terms unresolved; official input resolution unpublished |
| the encoder benchmark | depends on the above |
| Stage 3 intervention | **disabled by design**; prerequisites unmet, and §5 argues against it |
| anything on Kaggle | package only |
| anything touching `internal_test` | **locked** |
| anything touching Vietnam REPORT | **not read** |

## 4. Audit findings, with their limitations

### The one measured result — exploratory

Frozen baseline score, **Vietnam TUNE only** (n = 1,352):

| group | n | median score |
|---|---|---|
| Xpert-negative, no previous TB | 796 | **−0.3475** |
| Xpert-negative, previous TB | 417 | **+2.0509** |
| Xpert-positive, no previous TB | 90 | +3.1505 |
| Xpert-positive, previous TB | 49 | +3.0858 |

Among negatives the difference in medians is **+2.3984, 95% CI [1.9685, 3.1057]**
(10,000-replicate bootstrap, seed 20261021), and the score separates previous-TB from
other negatives at **AUROC 0.6957**.

**Among positives the same separation is at chance: AUROC 0.4973.**

### What it means, and what it does not

**New and useful:** the elevation is **specific to negatives**. Previous-TB history
carries no score signal among true positives. But previous-TB negatives (+2.05) overlap
heavily with all positives (≈+3.1), so **any suppression strong enough to push them down
would land in the positive distribution.** That is a measured argument for caution about
Stage 3, not a hunch.

**Limitations.** Exposed cohort; subgroup not prespecified; `past_tb` is **clinical
history, not a radiologist annotation** — no image-visible fibrosis may be inferred from
it; an association is not evidence the classifier uses any visual cue; **no causal claim**;
and the effect itself is **not novel** — Kagujje et al. 2022, Qin et al. 2024 and Codlin
et al. 2025 all report it, the last on this exact cohort.

### Structural findings

- **Cached RAD-DINO tokens cannot be reused** for CLEAR or RayDINO. Different
  architectures, resolutions and dimensions. Images must be re-encoded.
- **The `latent_tb` label mismatch is confirmed** (139 images: `tb_label=1`,
  `active_tb_label=0`). Both arms use `tb_label` identically so the comparison is not
  confounded by a label change.
- **Patch grids cannot match** (37 against an unknown grid from patch 16). The primary
  comparison is therefore matched **global** heads; regional is secondary via declared
  pooling.
- **No cohort can be called unseen** by RAD-DINO, CLEAR or RayDINO — all three have
  unknown checkpoint-selection exposure.

## 5. Measured results versus hypotheses

| | |
|---|---|
| **Measured** | the four subgroup medians, the +2.3984 [1.9685, 3.1057] difference, the 0.6957 and 0.4973 separations, the input inventory, the label semantics, the three licences |
| **Hypothesis, untested** | that chronic-abnormality concepts explain the elevation |
| **Hypothesis, untested** | that RayDINO changes discrimination in either direction |
| **Argued, not tested** | that suppression would harm true positives — argued *from* the measured overlap |

## 6. Blockers

| blocker | type | resolution |
|---|---|---|
| **Vietnam image licence** | **data rights** | Dryad names no licence. Ask the depositors (Codlin et al.) before uploading anything. **Blocks the concept stage** |
| **RayDINO weights gated** | access | request at `ai.meta.com/resources/models-and-libraries/raydino-downloads/` |
| **RayDINO redistribution to Kaggle** | licence | FAIR Noncommercial Research License — re-hosting permission **not established** |
| **RayDINO input resolution** | documentation | unpublished on the pages retrieved; the loader raises rather than guessing |
| **CLEAR commit hash** | reproducibility | none published; pin one |
| TBX11K redistribution terms | data rights | unverified, therefore blocked |

## 7. Kaggle-ready file list and the exact manual steps

1. Read `DATA_RIGHTS_AND_MODEL_LICENCES.md`. **Settle the Vietnam image licence first.**
2. Copy `manifest_template.json` → `manifest.json` and fill it honestly. **`unknown`
   blocks the concept stage by design.**
3. Create a **private** Kaggle dataset with `src/`, `configs/`, `manifest.json`.
4. Create a second **private** dataset with the data files in
   `KAGGLE_INPUT_CHECKLIST.md` §B — **strip the 901 `internal_test` rows from
   `provisional_c01.csv` first**, and include images only if §D is fully ticked.
5. New notebook: **GPU T4×2 or P100**, internet **on**, persistence **Variables and
   files**.
6. Run `00_preflight.ipynb`. **If it fails, stop.**
7. Run `01_concept_audit.ipynb` — the score-only analysis runs regardless; concept
   inference only if the manifest clears the licence.
8. For the benchmark: obtain RayDINO access, set `RAYDINO_WEIGHTS`, set its input
   resolution from the model card, then run `02_raydino_vs_raddino.ipynb`.
9. Leave `03_optional_targeted_ablation.ipynb` **disabled**.

Troubleshooting table: `README_KAGGLE.md`. Offline verification: `LOCAL_DRY_RUN.md`.

## 8. One recommended next step

**Email the Vietnam cohort depositors to establish the image licence.** It is one message,
it costs nothing, and it is the gate on the concept audit — the only stage whose
prerequisites are otherwise met, since CLEAR is Apache-2.0 and the images are already held
locally.

It is also the better use of effort than the benchmark. RayDINO needs a gated request, an
unresolved redistribution question, and an unpublished resolution; and from
`DECISION_RULES.md` it would have to clear +0.02 AUROC with an interval excluding zero and
no sensitivity loss in the previous-TB subgroup. Given that every intervention measured in
this project so far has moved the screening number by less than that, the prior on a
different frozen encoder clearing it is not high — and a null would be uninformative,
because the only evaluation cohort available is itself a published benchmark.

**No deployment recommendation follows from any of this, and nothing here is clinically
validated.**
