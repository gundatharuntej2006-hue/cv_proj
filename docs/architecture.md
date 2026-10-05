# System Architecture & Technical Specifications

## Pipeline Overview
The Explainable TB Screening CAD System is organized into 5 functional layers:

1. **Quality & Preprocessing Layer (`p2_gate`, `p1_data`, `p1_viewhead`)**:
   - Analyzes raw radiograph for motion blur (Laplacian variance), contrast (histogram spread), and OOD status.
   - Verifies posteroanterior (PA) positioning.
   - Extracts bounding crop around bilateral lung fields.

2. **Classification Layer (`p2_cls`)**:
   - Extracts medical visual tokens via RAD-DINO foundation model.
   - Multi-class head predicts probabilities across:
     - `HEALTHY` (normal lung parenchyma)
     - `SICK_NON_TB` (pneumonia, COVID-19, nodule, etc.)
     - `TB` (active pulmonary tuberculosis)

3. **Lesion Localisation Layer (`p3_det`)**:
   - D-FINE DETR-based detector outputs bounding boxes with confidence scores.
   - Identifies typical manifestation types: apical cavitations, upper lobe consolidations, miliary nodules.

4. **Visual Explainability Layer (`p3_xai`)**:
   - Grad-CAM heatmap generation from the final convolutional/attention layer.
   - Evaluated via Pointing Game Hit rate against radiologist bounding boxes.

5. **Statistical Triage & Delivery Layer (`p4_stats`, `p4_pipeline`, `p4_eval`, `p1_app`, `p3_ops`)**:
   - Calibrates raw logits using temperature scaling.
   - Applies Conformal Risk Control with tolerance $\alpha \le 0.10$ to assign `TB`, `REFER`, or `NOT_TB`.
   - Generates interactive UI and automated PDF reports.
