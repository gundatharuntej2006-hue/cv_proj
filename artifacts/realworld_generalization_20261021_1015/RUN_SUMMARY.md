# Run summary

**Directory:** `artifacts/realworld_generalization_20261021_1015/`
**Start 10:24 · end 11:05 local · ≈41 minutes**
**HEAD `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, unchanged start to end**

---

## Status table

| phase | status | key output |
|---|---|---|
| 0 — state and evidence audit | **complete** | `00_STATE_AND_EVIDENCE_AUDIT.md`, `tables/cohort_exposure_registry.csv` (12 rows) |
| 1 — hypotheses and identifiability | **complete** | `01_HYPOTHESES_AND_IDENTIFIABILITY.md`, `tables/available_training_label_structure.csv` |
| 2 — preregistered experiment | **complete; verdict INCONCLUSIVE** | `02_EXPERIMENT_PREREGISTRATION.md` (sha256 `f748b4fc…`), `03_EXISTING_DATA_EXPERIMENT_REPORT.md` |
| 3 — second cohort design | **complete (design only; data access is the blocker)** | `04_…PROTOCOL.md`, `05_SAMPLE_SIZE_AND_POWER.md`, `06_DATA_ACQUISITION_CHECKLIST.md`, `tables/data_source_shortlist.csv` |
| 4 — prior art and contribution | **complete** | `07_PRIOR_ART_AND_CONTRIBUTION_ASSESSMENT.md` |
| 5 — reproducibility and compute | **complete** | `08_REPRODUCIBILITY_AND_COMPUTE.md`, `tables/compute_audit.json` |
| 6 — synthesis and decision | **complete** | `FINAL_RESEARCH_DECISION.md` |

**Nothing was blocked.** Phase 3 produced a design rather than data because no qualifying
cohort exists to obtain — that is the finding, not a failure.

## Headline numbers

| | |
|---|---|
| **new diagnostic** | 35.0% of Vietnam negatives have documented past TB; median score +2.175 against −0.338 for other negatives |
| **the gap, localised** | AUROC 0.7811 against negatives without past TB; **0.6182** against those with it; Lunit 0.8170 → 0.7192 on the same split |
| **preregistered primary (arm X vs H)** | **+0.0074 [−0.0123, +0.0264] — INCONCLUSIVE**, minimum effect was +0.05 |
| **secondary, not promoted (arm L)** | +0.0292 [+0.0223, +0.0363] primary · +0.0141 [+0.0095, +0.0188] full cohort |
| **secondary (arm S)** | −0.0136 [−0.0352, +0.0073] — adding non-TB-abnormal negatives does not help |
| **deployed configuration** | **unchanged. Nothing adopted. Vietnam REPORT remains 0.7242** |

## Verification

| | |
|---|---|
| streaming loader vs committed `repool` | **100.000000% identical**, max abs diff 0.0 |
| baseline arm vs published Vietnam figure | 0.7245 vs 0.7240 (GPU nondeterminism, flagged) |
| preregistration hashed before scoring | yes, `f748b4fc1cad44e6…`, asserted by test |
| tests | **12 (this run) · 228 passed + 1 skipped (production) · 19 + 11 (prior run dirs)** |
| peak VRAM | **0.94 GB** of a 5.15 GB cap |

## Prior art

The past-TB finding **replicates** Kagujje et al., *Clin Infect Dis* 2022;76(3):e894–e901
(DOI 10.1093/cid/ciac679), verified by full-text retrieval: CAD4TB specificity 24.0% vs
60.3%, qXR 22.2% vs 61.8%, with and without previous TB. This run's addition is that the
deficit appears in **AUROC**, so it is a ranking failure rather than a threshold artefact.

## Safety

No commits, no pushes, no deletions. Production source, checkpoints, caches, predictions
and prior reports **unmodified**. The six tracked documentation files modified by the
preceding task were **preserved**. Five prior run directories **untouched**. TBX11K
`internal_test` **not read**. The OOD gate **not touched**. No background job started or
left running; the two long-lived `verdant-ai` uvicorn processes predate this session and
were deliberately left alone.

## Decision

**NO-GO on further model changes before representative data.** Next action: approach the
Kagujje / Muyoyeta group at CIDRZ Lusaka for a second screening cohort of 2,500–4,000
participants with universal microbiological confirmation. Full reasoning and the three
narrow exceptions in `FINAL_RESEARCH_DECISION.md` §7–8.
