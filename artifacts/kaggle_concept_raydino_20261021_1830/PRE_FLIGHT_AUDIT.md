# Stage 0 — Pre-flight audit

**Run:** `artifacts/kaggle_concept_raydino_20261021_1830/` · 2026-10-21T19:22 +05:30
**Git HEAD:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a`
**Read-only.** No model fitted, no image encoded, no weight downloaded.
Machine-readable: `tables/preflight_audit.json`, `tables/input_inventory.csv`.

---

## 1. The locked partition

**TBX11K `internal_test` — 901 rows — was never opened.** Its row count was read from
`provisional_c01.csv`'s `split` column and the rows were dropped before any other use, as
asserted in code. No image path, label, ID or score from it was loaded, joined, logged or
written to any manifest. `guards.assert_path_allowed` raises on any path containing the
token, and five tests cover that behaviour.

## 2. Input inventory

Full table with checksums: `tables/input_inventory.csv`.

| kind | asset | status |
|---|---|---|
| images | `data/processed/v1/vietnam_img512` | **present** — 2,708 PNG, 346 MB, named `<pid>.png` |
| images | `data/processed/v1/vietnam_img512_lungs` | present — 2,708 PNG, 149 MB |
| images | TBX11K `health` / `sick` / `tb` | present — 3,800 / 3,800 / 800, ≈2.9 GB |
| metadata | Vietnam Dryad participant table | present — 2,708 rows |
| splits | `vietnam_tune_report_split.json` | present — TUNE 1,352, REPORT 1,356, **overlap 0** |
| splits | `provisional_c01.csv` | present — usable: train 5,941, development 659, calibration 899 |
| labels | `labels_tbx11k_trainval_v1.csv` | present — 8,400 rows |
| scores | frozen baseline per-case Vietnam scores | present — 2,708 rows |
| features | `regional_vietnam_lungs_g12_v1.npz` | present — RAD-DINO 12×12×768, **not convertible to another encoder** |

**Images are available locally.** That was not a given and it unblocks the technical path
for both stages. What blocks them is licensing, not data.

## 3. Baseline checkpoint and what the cached artifacts actually contain

The frozen baseline is RAD-DINO ViT-B/14, revision
`110cbc18d5133582e320b43d53bf5c44e410c936`, 518 px, 37×37 patch grid pooled to 12×12,
with a per-region linear head max-pooled over 144 regions.

The cached artifacts contain **RAD-DINO regional tokens (12×12×768 float16)**, **RAD-DINO
global embeddings (768-d)** and **per-case scores**. They contain **no image pixels**.

**Therefore the cached features cannot be reused for CLEAR or RayDINO.** Both are
different encoders with different architectures, input resolutions and embedding
dimensions. Images must be re-encoded from scratch for each. This is recorded because
assuming otherwise would silently invalidate the benchmark.

## 4. Target-label semantics

| category | n | `tb_label` | `active_tb_label` |
|---|---|---|---|
| healthy | 3,800 | 0 | 0 |
| sick_non_tb | 3,800 | 0 | 0 |
| active_tb | 631 | 1 | 1 |
| **latent_tb** | **139** | **1** | **0** |
| active_latent_tb | 30 | 1 | 1 |

**The mismatch is real and confirmed:** `latent_tb` is positive under `tb_label` while its
own `active_tb_label` says 0. The frozen baseline trains on `tb_label`.

**Decision:** both encoder arms use `tb_label` identically. The mapping is a **documented
limitation of the reference pipeline**, not something to fix in one arm — correcting it in
one arm only would confound the encoder comparison with a label change.
`configs/target_label.yaml` records it, and `guards.assert_target_mapping` refuses to
train on an undefined or overlapping mapping.

An active-only alternative was measured in an earlier run as a prespecified secondary arm
(+0.0292 on a past-TB subgroup endpoint, below that run's declared threshold). It is
**not adopted and not mixed into this benchmark.**

## 5. Encoder provenance and cohort exposure

| encoder | licence (code / weights) | training data | exposure to TBX11K, Shenzhen, Montgomery, DA/DB, Vietnam |
|---|---|---|---|
| **RAD-DINO** | MIT / MIT | MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX (882,775) | **UNKNOWN** — the card says the checkpoint was selected by linear probing on unnamed evaluation validation sets |
| **CLEAR** | Apache-2.0 / Apache-2.0 | MIMIC-CXR, CheXpert-Plus, ReXGradient (~0.87M pairs, 239,391 patients) | **UNKNOWN** — no TB benchmark named, but no evaluation set named either |
| **RayDINO** | Apache-2.0 (DINOv2) / **FAIR Noncommercial Research License** | 873k chest X-rays | **UNKNOWN** |

**No cohort may be described as unseen by any of these encoders.** Verified from official
sources on 2026-10-21; full detail in `DATA_RIGHTS_AND_MODEL_LICENCES.md`.

## 6. Hard constraints — all pass

| constraint | status |
|---|---|
| `internal_test` never opened | **PASS** — count only, dropped before use, guarded in code, 5 tests |
| Vietnam REPORT not read | **PASS** — Stage 1 restricted to TUNE; REPORT never loaded |
| Vietnam TUNE marked exploratory | **PASS** — labelled in every output |
| no production file modified | **PASS** — everything written to the new run directory |
| no commit, push, delete, background job, credential read | **PASS** |
| no data uploaded anywhere | **PASS** — code-only package |
| model licences verified from official sources | **PASS** — three components, recorded |
| no fibrosis inferred from `past_tb` | **PASS** — stated explicitly in every output |
| no causal claim from probes or correlations | **PASS** |
| no clinical-readiness claim | **PASS** |
| manifest written before any fit | **PASS** — `RUN_PROTOCOL.md`, `DECISION_RULES.md` hashed in `MANIFEST.sha256` |
| shapes, IDs, labels asserted | **PASS** — 29 synthetic tests |

## 7. Blockers that stop execution here

1. **Vietnam image licence — BLOCKED.** The Dryad record names no specific licence, so
   uploading images or the patient table to Kaggle is not permitted. **Unknown blocks.**
2. **RayDINO weights — GATED**, and whether re-hosting them on Kaggle is permitted under
   the FAIR Noncommercial Research License is **unresolved**. The user must request access.
3. **CLEAR has no published commit hash.** Pin one before relying on its concept bank.
4. **RayDINO's official input resolution is unpublished on the pages retrieved.** The
   loader raises rather than guessing, since 16 does not divide 518.

**Consequence: this run prepares a code-only package and executes only the safe
score-based analysis. No concept score, no encoder benchmark and no Kaggle run exists.**
