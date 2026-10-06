# Explainable TB Screening from Chest X-Rays — Work Plan (4 Workstreams)

[![CI](https://github.com/gundatharuntej2006-hue/cv_proj/actions/workflows/ci.yml/badge.svg)](https://github.com/gundatharuntej2006-hue/cv_proj/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**BAI506 Mini Project, BMSIT&M 2026–27 · Team of 4 · 12 weeks · RTX 4050 6 GB laptop + Kaggle/Colab T4**  
*Department of Artificial Intelligence & Machine Learning, BMS Institute of Technology and Management*  
**Under the Guidance of:** Dr. Niranjanamurthy M, Dept. of AI & ML

> This README is the team's operating plan: who builds what, which files connect the workstreams, and how results stay honest.

| Person | Name | GitHub Handle | Workstream |
|---|---|---|---|
| **Person 1** | Gunda Tharun Tej | `@gundatharuntej2006-hue` (`@tb-p1`) | Data, preprocessing, segmentation → dashboard |
| **Person 2** | Atul Dhull | `@tb-p2` | RAD-DINO classification, quality/OOD gate → baselines, report assembly |
| **Person 3** | Ritika Girish Kulkarni | `@tb-p3` | D-FINE detection, Grad-CAM → PDF, logging, latency |
| **Person 4** | Kunal | `@tb-p4` | Statistics, calibration, conformal triage, pipeline, evaluation, governance |

---

## 1. Project Summary

1. **Input:** Frontal chest X-ray (DICOM or PNG), decoded, converted to grayscale and standardized to 512×512.
2. **Gate:** Image-quality check plus a RAD-DINO kNN Out-Of-Distribution (OOD) score. Rejected images route to manual review.
3. **Analysis:** Pretrained TorchXRayVision lung segmentation with a sanity check, then lung crop with 10% margin (fallback to full image if check fails). Frozen RAD-DINO extractor with linear head outputs TB probability; D-FINE detects single-class TB lesions.
4. **Decision:** Temperature scaling (fitted on development split), then conformal lower threshold ($t_{\text{lower}}$) and precision-constrained upper threshold ($t_{\text{upper}}$) fitted on calibration split to yield **TB / REFER / NOT TB**. The detector link rule upgrades suspect `NOT TB` cases with detected lesions to `REFER`.
5. **Output:** Worklist sorted by calibrated TB probability with Grad-CAM activation overlays, lesion boxes, per-patient PDF export, and structured JSONL prediction logs. Every metric carries sample size $n$ and 95% CIs. **Target:** Sensitivity $\ge 90\%$ and Specificity $\ge 70\%$ (Screen-Positive = TB $\cup$ REFER).

### "Prototype Done" Definition of Done
- `make demo` starts the Streamlit dashboard on laptop. A mixed batch (DICOM, PNG, corrupt file, lateral view, non-medical image) produces a worklist with every Section 11 field, a PDF per case, and one log line per case.
- All 23 decision files (D01–D23) are locked, the no-test-tuning gate passes, and the internal test split is evaluated **once**. Tables 12.1–12.8 exist with $n$ and CIs.
- One challenge submission logged using Section 4.6 category mapping fixed in D21. External validation (§13) and comparative experiments (§14) complete.
- Latency E6 measured and reported against $< 2\text{ s}$ (GPU) and $< 10\text{ s}$ (CPU).
- Report, slides, and backup demo video finished; repository tagged `v-freeze`.

---

## 2. Architecture & Pipeline Flow

```
[Raw CXR (DICOM/PNG)]
         │
         ▼
[P2 Quality Gate & kNN OOD Check] ──(Reject)──> [Manual Review List (REJECTED_GATE)]
         │ (Pass)
         ▼
[P1 Lung Segmentation & Sanity Crop] (TorchXRayVision + 10% Margin Crop)
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
[P2 Classification]       [P3 Lesion Detection]
(RAD-DINO Frozen Head)    (D-FINE Bounding Boxes)
    │                             │
    ▼                             │
[P3 Explainability]              │
(Grad-CAM Patch Grid)             │
    │                             │
    └──────────────┬──────────────┘
                   ▼
        [P4 Calibration & Triage]
        (Temperature Scaling + Conformal Risk Control <= 10% FNR + Link Rule)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[P1 Streamlit Dashboard] [P3 ReportLab PDF Export]
```

---

## 3. Repository Structure & Workstream Ownership

Rule: **You edit only paths you own.** You may import another person's public API (`api.py`) or read their contract files, but you never edit them.

```
cv_proj/
├── README.md                              P4 (steward; any change needs all-4 review)
├── .github/
│   ├── CODEOWNERS                         P4
│   └── workflows/ci.yml                   P3 (pytest + contract validators + ownership check)
├── .gitignore  .gitattributes  .pre-commit-config.yaml   P1 (.gitattributes = Git LFS rules)
├── environment.yml  pyproject.toml  Makefile             P1 (Makefile includes mk/pN.mk)
├── mk/p1.mk  mk/p2.mk  mk/p3.mk  mk/p4.mk                each owner
├── requirements/base.txt                  P1
├── requirements/p1.txt … p4.txt           each owner (own extra deps)
├── third_party/D-FINE/                    P3 (git submodule, pinned commit)
│
├── docs/
│   ├── methodology.md  proposal.md        frozen inputs
│   ├── p1_datasets.md  p1_storage_budget.md  p1_usability.md  p1_demo_script.md      P1
│   ├── p2_vram_feasibility.md  p2_ood_sources.md                                      P2
│   ├── p3_dfine_feasibility.md  p3_kaggle_runbook.md  p3_gradcam_review.md            P3
│   └── p4_codabench_log.md  p4_gate_audit.md                                          P4
│
├── contracts/                             P4 steward · FROZEN at tag contracts-v1 (end W1)
│   ├── README.md                          contract index C01–C18, R1–R4
│   ├── schemas/                           formal JSON Schema for C01, C14, C16, C18
│   ├── specs/                             specifications for remaining contracts
│   ├── case_result.schema.json            C14
│   ├── decision.schema.json               C16
│   └── ccr/CCR-NNN.md                     contract change requests
│
├── decisions/                             one JSON per locked decision; owner = filename prefix
│   ├── README.md                          P4 (registry index D01–D23)
│   ├── p1_*.json                          P1  (D01–D05)
│   ├── p2_*.json                          P2  (D06–D15)
│   ├── p3_*.json                          P3  (D16–D20)
│   └── p4_*.json                          P4  (D21–D23)
│
├── data/                                  gitignored except data/splits/
│   ├── raw/tbx11k/ raw/shenzhen/ raw/montgomery/ raw/fallback/                 P1
│   ├── raw/ood/                                                                P2
│   ├── splits/tbx11k_splits_v1.csv  splits/external_v1.csv   (committed)       P1
│   ├── splits/ood_heldout_v1.csv                             (committed)       P2
│   ├── processed/v1/                      C04 store, masks, C05                P1
│   └── boxes/v1/                          C06 COCO files                       P1
│
├── src/
│   ├── tbcore/        P4  validators, sealed-split guard, seed, paths, version stamps, disclaimer
│   ├── p1_data/       P1  ingest, anonymise, splits, TXRV seg, sanity, crop, boxes; api.py = R1
│   ├── p1_app/        P1  Streamlit dashboard, worklist, threshold slider
│   ├── p1_viewhead/   P1  (extension) frontal/lateral head
│   ├── p2_cls/        P2  RAD-DINO embeddings, heads, EfficientNet-B3; api.py = R2 (classify)
│   ├── p2_gate/       P2  image-quality check, kNN OOD; api.py = R2 (gate)
│   ├── p3_det/        P3  D-FINE adapter, training configs, detection metrics; api.py = R3 (detect)
│   ├── p3_xai/        P3  Grad-CAM for ViT, localisation metrics, overlay; api.py = R3 (explain)
│   ├── p3_ops/        P3  Kaggle→Drive checkpointing, PDF, prediction logger, latency E6
│   ├── p4_stats/      P4  Wilson/bootstrap CIs, ECE, temperature, conformal, upper threshold, link
│   ├── p4_pipeline/   P4  run_case / run_batch → CaseResult; api.py = R4
│   └── p4_eval/       P4  no-test-tuning gate, tables 12.1–12.8, external, §14, challenge bundle
│
├── tests/tbcore/ tests/p1_*/ tests/p2_*/ tests/p3_*/ tests/p4_*/     owner by prefix
├── notebooks/p1/ p2/ p3/ p4/                                         owner by folder
├── artifacts/p1/ p2/ p3/ p4/          owner by folder (small files in LFS, large on Drive + SHA)
├── mock/                              gitignored; each producer writes only its own contracts
├── runs/                              gitignored; CaseResult outputs (P4)
├── logs/predictions.jsonl             gitignored; C15 (P3)
└── report/
    ├── main.md  abstract/intro/conclusion  slides/                   P2
    ├── p1_*.md   p2_*.md   p3_*.md   p4_*.md                         each owner
    └── demo_video.md                                                 P1
```

---

## 4. Interface Contracts (C01–C18) & Runtime APIs (R1–R4)

| Contract | Target File / Format | Producer | Primary Consumers |
|---|---|---|---|
| **C01** | `data/splits/tbx11k_splits_v1.csv` | P1 | P2, P3, P4, `tbcore.guard` |
| **C02** | `data/splits/external_v1.csv` | P1 | P2, P4 |
| **C03** | `data/splits/ood_heldout_v1.csv` | P2 | P4, P1 |
| **C04** | `data/processed/v1/manifest.parquet` + PNGs | P1 | P2, P3 |
| **C05** | `data/processed/v1/lung_roi.parquet` | P1 | P2, P3, P4 |
| **C06** | `data/boxes/v1/coco_{split}_m{margin}.json` | P1 | P3, P4 |
| **C07** | `artifacts/p2/emb/radino_{dataset}_{variant}_v1.npz` | P2 | P2, P1 |
| **C08** | `artifacts/p2/scores/scores_{dataset}_{split}_v1.parquet` | P2 | P4 |
| **C09** | `artifacts/p2/gate/gate_{dataset}_{split}_v1.parquet` | P2 | P4 |
| **C10** | `artifacts/p3/det/detections_{dataset}_{split}_v1.json` | P3 | P4 |
| **C11** | `artifacts/p3/xai/...` (NPY + Parquet) | P3 | P4, P1 |
| **C12** | `artifacts/p4/triage/triage_{dataset}_{split}_v1.parquet` | P4 | P1, P2, P3 |
| **C13** | `artifacts/p4/triage/operating_curve_calibration_v1.parquet`| P4 | P1 |
| **C14** | `runs/{run_id}/{case_id}.json` (`CaseResult`) | P4 | P1, P3 |
| **C15** | `logs/predictions.jsonl` | P3 | P4, P1 |
| **C16** | `decisions/pN_*.json` (D01–D23) | Each Owner | P4 Gate, All |
| **C17** | `artifacts/p2/dedup/near_dup_candidates_v1.csv` | P2 | P1 |
| **C18** | `artifacts/p4/tables/t{section}_{name}.csv` | P4 | P1, P2, P3 |

### Runtime Python Entry Points
- **R1 (`p1_data.api`)**: `standardise(path) -> StdImage`, `lung_roi(img512) -> Roi`
- **R2 (`p2_cls.api`, `p2_gate.api`)**: `gate(img512) -> GateResult`, `classify(img) -> ClsResult`, `classifier_module() -> Module`
- **R3 (`p3_det.api`, `p3_xai.api`, `p3_ops.api`)**: `detect(img, crop_box) -> DetResult`, `explain(img, crop_box) -> heatmap`, `render_overlay(...) -> png_path`, `make_pdf(case_result) -> pdf_path`, `log_case(case_result, operating_point)`
- **R4 (`p4_pipeline.api`)**: `run_case(path) -> CaseResult`, `run_batch(paths) -> list[CaseResult]`, `triage(p_tb_cal, n_lesions, t_lower, t_upper) -> category`

---

## 5. Quickstart & Installation

### Option A: Using Conda (Recommended)
```bash
git clone https://github.com/gundatharuntej2006-hue/cv_proj.git
cd cv_proj
conda env create -f environment.yml
conda activate tb-screening
pip install -e .
```

### Option B: Using Pip & Make
```bash
make install-base

# Person-specific dependencies:
make install-p1   # Data & Streamlit Dashboard
make install-p2   # RAD-DINO & Quality Gate
make install-p3   # D-FINE & Explainability
make install-p4   # Statistics, Conformal Triage & Governance

# Or install all:
make install-all
```

---

## 6. Running Verification & Commands

```bash
# 1. Run complete pytest test suite (tbcore + p1/p2/p3/p4)
python -m pytest

# 2. Verify all 23 decision files with No-Test-Tuning Gate
python -m p4_eval.gate check

# 3. Generate mock contracts and 7 CaseResult test fixtures
python -m tbcore.mock_p4

# 4. Generate C18 evaluation tables (Tables 12.1-12.7, External, Comparisons)
make eval-tables

# 5. Launch Streamlit Clinician Dashboard
streamlit run src/p1_app/dashboard.py
```

---

## 7. Governance & No-Test-Tuning Gate Rules

1. **Pre-Registered Criteria:** Every decision JSON file in `decisions/` has its criterion recorded prior to running experiments.
2. **Single-Use Gate Tokens:** Access to sealed test splits (`internal_test`, `challenge_test`, `external_test`, `ood_eval`) is strictly governed by `p4_eval.gate`, which validates all D01–D23 decisions before issuing single-use tokens.
3. **Traceability:** Every metric row in evaluation tables carries $n_{\text{tb}}$, $n_{\text{non\_tb}}$, and 95% Wilson / Bootstrap confidence intervals.
4. **Scope Honesty:**
   > **Disclaimer:** Decision support for TB triage only, not a diagnosis. Validated only on public research datasets; not prospectively clinically validated.
