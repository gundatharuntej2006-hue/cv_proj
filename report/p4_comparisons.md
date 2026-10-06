# Comparative Experiments Summary (Section 14, Table 14)

**Author:** Person 4 (`@tb-p4`)  
**Artifacts:** `artifacts/p4/tables/t14_comparisons.csv`

---

## 1. Summary of 9 Core Scientific Questions

| Question | Options Compared | Deciding Metric | Choice Locked | Sealed Test Result (95% CI) |
|---|---|---|---|---|
| 1. Backbone | RAD-DINO vs EfficientNet-B3 | ROC-AUC | RAD-DINO (D11/D14) | 0.942 [0.915, 0.968] vs 0.912 [0.880, 0.941] |
| 2. Output Head | Binary vs 3-Class | Sens @ 70% Spec | Binary (D11) | 0.920 [0.850, 0.958] vs 0.895 [0.820, 0.940] |
| 3. Lung Cropping | Crop (margin 0.10) vs Full | ROC-AUC | Crop m=0.10 (D09) | 0.942 [0.915, 0.968] vs 0.921 [0.889, 0.949] |
| 4. CLAHE | Off vs On | Sens @ 70% Spec | CLAHE Off (D08) | 0.920 [0.850, 0.958] vs 0.905 [0.832, 0.947] |
| 5. Calibration | Uncalibrated vs Scaled | ECE (10 bins) | $T=1.37$ (D22) | 0.034 [0.018, 0.056] vs 0.089 [0.062, 0.118] |
| 6. Triage Lower | Conformal vs ROC Youden | Sensitivity | Conformal (D23) | 0.920 [0.850, 0.958] vs 0.890 [0.814, 0.938] |
| 7. Link Rule | With vs Without Link | Sens / Refer Rate | Link Enabled (D23) | Sens +1.0% (93.0%), Refer +2.4% |
| 8. OOD Gate | $k=5$ vs $k=1$ | OOD Rejection | $k=5$ (D13) | 96.2% [94.2%, 97.6%] vs 93.4% [91.1%, 95.2%] |
| 9. Detector Size | 512 vs 640 | AP@0.50 | 512x512 (D16) | 0.421 [0.365, 0.482] (512 chosen for latency/VRAM) |
