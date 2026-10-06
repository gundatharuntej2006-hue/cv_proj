# Decisions Registry (D01–D23)

**Steward:** Person 4 (`@tb-p4`)  
**Governance:** Every decision is formally recorded in a versioned JSON file (Schema C16) and locked before evaluating sealed test sets.

---

## Decision Index

| ID | File | Owner | Title | Pre-registered Criterion | data_used | Status |
|---|---|---|---|---|---|---|
| **D01** | `p1_splits.json` | P1 | Dataset Splits & Stratification | Seed, 5-category stratification, exact split counts | none | LOCKED |
| **D02** | `p1_dedup_resolution.json` | P1 | Cross-Split Near-Duplicate Resolution | Pair review; removal from later split in H1 order | images_only_all_splits | LOCKED |
| **D03** | `p1_seg_sanity_ranges.json` | P1 | Lung Segmentation Sanity Ranges | Area & lung box size in [1st, 99th] percentile of dev split | development | LOCKED |
| **D04** | `p1_external_source.json` | P1 | External Cohort Data Sources | NLM Shenzhen & Montgomery cohorts | none | LOCKED |
| **D05** | `p1_licences.json` | P1 | Dataset Licences (P1) | TBX11K CC BY 4.0, NLM terms, TXRV Apache-2.0 | none | LOCKED |
| **D06** | `p2_dedup_threshold.json` | P2 | Near-Duplicate Cosine Threshold | Cosine distance tau=0.98 on L2-normalized embeddings | images_only_all_splits | LOCKED |
| **D07** | `p2_quality_rules.json` | P2 | Image Quality Gate Rules | Blank, exposure, and crop thresholds on synthetic tests | development | LOCKED |
| **D08** | `p2_clahe.json` | P2 | CLAHE Preprocessing On/Off | Sensitivity at 70% specificity on dev split | train, development | LOCKED |
| **D09** | `p2_crop_margin.json` | P2 | Lung Box Crop Margin Selection | Margin in {0.0, 0.05, 0.10, 0.15} by dev sensitivity | train, development | LOCKED |
| **D10** | `p2_training_recipe.json` | P2 | Imbalance Mitigation Recipe | Weighted sampler vs logit-adjusted loss | train, development | LOCKED |
| **D11** | `p2_head.json` | P2 | Primary Classification Head | Binary vs 3-class head on dev sensitivity | train, development | LOCKED |
| **D12** | `p2_effnet_baseline.json` | P2 | EfficientNet-B3 Baseline Weights | Dev AUC early stopping checkpoint | train, development | LOCKED |
| **D13** | `p2_ood_gate.json` | P2 | kNN OOD Gate Parameters | k=5 mean cosine distance, 95% dev pass threshold | train, development | LOCKED |
| **D14** | `p2_model_registry.json` | P2 | Frozen Model Registry Weights | SHA-256 hashes of all models entering sealed test | none | LOCKED |
| **D15** | `p2_licences.json` | P2 | Model & OOD Licences (P2) | RAD-DINO research licence, CheXpert research terms | none | LOCKED |
| **D16** | `p3_det_input_size.json` | P3 | Detector Input Resolution | 512x512 vs 640x640 by dev AP@0.50 | train, development | LOCKED |
| **D17** | `p3_det_checkpoint.json` | P3 | D-FINE Training Checkpoint | Best dev AP@0.50 checkpoint SHA | train, development | LOCKED |
| **D18** | `p3_det_conf_threshold.json` | P3 | Detector Confidence Threshold | Lowest thr with <=5% non-TB dev images with boxes | development | LOCKED |
| **D19** | `p3_gradcam_config.json` | P3 | Grad-CAM ViT Layer & Transform | Last block attention, patch reshape, crop mapping | development | LOCKED |
| **D20** | `p3_licences.json` | P3 | Detector & XAI Licences (P3) | D-FINE Apache-2.0, pytorch-grad-cam MIT | none | LOCKED |
| **D21** | `p4_analysis_plan.json` | P4 | Statistical Analysis Protocol | Alpha=0.10, PPV=80%, Wilson & Bootstrap CIs, Section 4.6 | none | LOCKED |
| **D22** | `p4_temperature.json` | P4 | Temperature Scaling Parameter | Minimise NLL on dev split (T=1.37) | development | LOCKED |
| **D23** | `p4_triage_thresholds.json` | P4 | Triage Decision Thresholds | Conformal lower (t=0.18) & Precision upper (t=0.83) on cal | calibration | LOCKED |
