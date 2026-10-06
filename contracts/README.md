# Interface Contracts Registry (C01–C18) & Runtime API (R1–R4)

**Steward:** Person 4 (`@tb-p4`)  
**Status:** FROZEN at tag `contracts-v1` (End of Week 1)

---

## 1. Governance & Contract Rules

1. **Strict Data Boundary:** Team members exchange information and data artifacts **only** through these versioned contracts (C01–C18) and frozen Python entry points (R1–R4). Direct editing of another owner's folder is strictly forbidden by CODEOWNERS and CI checks.
2. **Single Producer:** Every contract has exactly one owner/producer. Consumers may only read contracts.
3. **Change Control (CCR):** Any modification after freeze requires a formal Contract Change Request (`contracts/ccr/CCR-NNN.md`) approved by both the producer and affected consumer. Changes to C01, C14, C16, C18 or governance require all 4 approvals.
4. **Mock Parity:** Mocks for every contract exist under `mock/` with identical relative paths and are accessible when `TB_MOCK=1`.
5. **Sealed Split Protection:** Access to `internal_test`, `challenge_test`, `external_test`, and `ood_eval` requires a single-use token from the P4 Gate.

---

## 2. File Contracts Index (C01–C18)

| Contract | Path / Artifact | Producer | Consumers | Format | Schema / Spec |
|---|---|---|---|---|---|
| **C01** | `data/splits/tbx11k_splits_v1.csv` | P1 | P2, P3, P4, `tbcore.guard` | CSV | `schemas/c01_splits.schema.json` |
| **C02** | `data/splits/external_v1.csv` | P1 | P2, P4 | CSV | `specs/c02_external.md` |
| **C03** | `data/splits/ood_heldout_v1.csv` | P2 | P4, P1 | CSV | `specs/c03_ood.md` |
| **C04** | `data/processed/v1/...` | P1 | P2, P3 | PNG + Parquet | `specs/c04_processed_store.md` |
| **C05** | `data/processed/v1/lung_roi.parquet` | P1 | P2, P3, P4 | Parquet | `specs/c05_lung_roi.md` |
| **C06** | `data/boxes/v1/coco_{split}_m{margin}.json` | P1 | P3, P4 | JSON (COCO) | `specs/c06_coco_boxes.md` |
| **C07** | `artifacts/p2/emb/radino_{dataset}_{variant}_v1.npz` | P2 | P2, P1 | NPZ | `specs/c07_embeddings.md` |
| **C08** | `artifacts/p2/scores/scores_{dataset}_{split}_v1.parquet` | P2 | P4 | Parquet | `specs/c08_classifier_scores.md` |
| **C09** | `artifacts/p2/gate/gate_{dataset}_{split}_v1.parquet` | P2 | P4 | Parquet | `specs/c09_gate_scores.md` |
| **C10** | `artifacts/p3/det/detections_{dataset}_{split}_v1.json` | P3 | P4 | JSON | `specs/c10_detections.md` |
| **C11** | `artifacts/p3/xai/{model_id}/{split}/{image_id}.npy` | P3 | P4, P1 | NPY + Parquet | `specs/c11_xai_heatmaps.md` |
| **C12** | `artifacts/p4/triage/triage_{dataset}_{split}_v1.parquet` | P4 | P1, P2, P3 | Parquet | `specs/c12_triage.md` |
| **C13** | `artifacts/p4/triage/operating_curve_calibration_v1.parquet` | P4 | P1 | Parquet | `specs/c13_operating_curve.md` |
| **C14** | `runs/{run_id}/{case_id}.json` | P4 | P1, P3 | JSON | `case_result.schema.json` |
| **C15** | `logs/predictions.jsonl` | P3 | P4, P1 | JSONL | `specs/c15_prediction_log.md` |
| **C16** | `decisions/pN_*.json` | All | P4 Gate, All | JSON | `decision.schema.json` |
| **C17** | `artifacts/p2/dedup/near_dup_candidates_v1.csv` | P2 | P1 | CSV | `specs/c17_near_duplicates.md` |
| **C18** | `artifacts/p4/tables/t{section}_{name}.csv` | P4 | P1, P2, P3 | CSV | `schemas/c18_tables.schema.json` |

---

## 3. Runtime Python APIs (R1–R4)

- **R1 (`p1_data.api`)**: Standardisation & Lung ROI cropping.
  - `standardise(path: str | Path) -> StdImage`
  - `lung_roi(img512: np.ndarray) -> Roi`
- **R2 (`p2_cls.api`, `p2_gate.api`)**: Quality Gate & RAD-DINO Classifier.
  - `gate(img512: np.ndarray) -> GateResult`
  - `classify(img_for_analysis: np.ndarray) -> ClsResult`
  - `classifier_module() -> torch.nn.Module`
- **R3 (`p3_det.api`, `p3_xai.api`, `p3_ops.api`)**: D-FINE Detection, Grad-CAM, PDF, Logger.
  - `detect(img_for_analysis: np.ndarray, crop_box: list[int]) -> DetResult`
  - `explain(img_for_analysis: np.ndarray, crop_box: list[int]) -> np.ndarray`
  - `render_overlay(img512: np.ndarray, heatmap: np.ndarray, boxes: list) -> Path`
  - `make_pdf(case_result: dict | CaseResult) -> Path`
  - `log_case(case_result: dict | CaseResult, operating_point: str)`
- **R4 (`p4_pipeline.api`, `p4_stats.api`)**: Pipeline Orchestrator & Triage.
  - `run_case(path: str | Path) -> CaseResult`
  - `run_batch(paths: list[str | Path]) -> list[CaseResult]`
  - `triage(p_tb_cal: float, n_lesions: int, t_lower: float, t_upper: float) -> str`
