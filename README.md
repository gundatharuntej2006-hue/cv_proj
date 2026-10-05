# Explainable TB Screening from Chest X-Rays with Lesion Localisation for Triage

[![CI](https://github.com/gundatharuntej2006-hue/cv_proj/actions/workflows/ci.yml/badge.svg)](https://github.com/gundatharuntej2006-hue/cv_proj/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Department of Artificial Intelligence & Machine Learning  
**BMS Institute of Technology and Management (BMSIT&M)**  
*Mini Project Phase – 0 & 1 (Course: BAI506)*  
**Under the Guidance of:** Dr. Niranjanamurthy M, Dept. of AI & ML

---

## 1. Executive Summary

Tuberculosis (TB) remains one of the world's deadliest infectious killers, with over **25% of global cases occurring in India**. Screening camps frequently employ portable digital X-ray units, generating volumes that overwhelm the availability of radiologists (~1 radiologist per 100,000 people in India). 

While deep neural networks can detect chest abnormalities, real-world clinical deployment fails when algorithms cannot distinguish pulmonary TB from confounding non-TB pathologies (e.g., bacterial pneumonia, COVID-19, COPD) or when predictions lack verifiable localization.

This project delivers an **end-to-end, explainable Computer-Aided Detection (CAD) and Triage System** compliant with the **World Health Organization (WHO) Target Product Profile (TPP)** recommendations:
- **Triage Decision:** Conformal sorting into `TB`, `REFER` (indeterminate / high uncertainty), or `NOT TB`.
- **Target Performance:** Sensitivity >= 90%, Specificity >= 70%, with Conformal Miss Rate (FNR) <= 10%.
- **Latency Requirement:** <= 2.0 seconds per radiograph on edge/laptop GPU.
- **Explainability:** Lesion-level bounding boxes (D-FINE detector) paired with Grad-CAM activation heatmaps validated against radiologist ground truth.

---

## 2. Project Architecture & Pipeline Flow

```
[Raw CXR (DICOM/PNG)]
         │
         ▼
[P2 Quality Gate & OOD Check] ──(Reject)──> [Clinical Rejection Report]
         │ (Pass)
         ▼
[P1 Viewhead & Lung Preprocessing] (PA verification + Lung Crop)
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
[P2 Classification]       [P3 Lesion Detection]
(RAD-DINO 3-Class)        (D-FINE Bounding Boxes)
    │                             │
    ▼                             │
[P3 Explainability]              │
(Grad-CAM Heatmap)                │
    │                             │
    └──────────────┬──────────────┘
                   ▼
        [P4 Calibration & Triage]
        (Conformal Risk Control <= 10% FNR)
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[P1 Clinician UI]   [P3 PDF Triage Report]
```

---

## 3. Team Responsibilities & Branch Ownership

The repository enforces strict module ownership to ensure four team members develop concurrently without merge collisions:

| Person | Role | Core Modules & Scope | Assigned Branch |
| :--- | :--- | :--- | :--- |
| **Tharun** ** | **P1** | Data ingestion, preprocessing, lung segmentation, viewhead classifier, Streamlit clinician dashboard | `p1` |
| **Atul** | **P2** | RAD-DINO 3-class classification (Healthy / Sick Non-TB / TB), Quality Gate, OOD detection, baseline wrappers | `p2` |
| **Ritika** | **P3** | D-FINE TB lesion detector, Grad-CAM attribution, pointing-game evaluation, PDF report export, runtime profiling | `p3` |
| **Kunal** | **P4** | Probability calibration (Temperature/Platt), Conformal Risk Control triage, end-to-end pipeline orchestration, bootstrap statistical CI evaluation, governance | `p4` |

*Shared modules (`src/tbcore/`, `contracts/`, `decisions/`, `README.md`) require cross-team review.*

---

## 4. Quickstart & Installation

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
make install-p1   # Tharun
make install-p2   # Atul
make install-p3   # Ritika
make install-p4   # Kunal

# Or install everything:
make install-all
```

---

## 5. Running Tests & Verification

```bash
# Run complete test suite
make test

# Verify all JSON Schema contracts
make test-contracts

# Verify file ownership and branch boundary integrity
make verify-ownership

# Launch the Streamlit Clinician Dashboard
make run-app
```
