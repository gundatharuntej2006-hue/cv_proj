# P3 report sections: number checklist (for P4's review)

Every number in report/p3_detector.md, p3_explainability.md, p3_operations.md and p3_limitations.md, with its
source. "Design" = a fixed setting or rule (not a measured result). Status on 2026-10-09.

## Numbers present now

| Number | Meaning | Source |
|---|---|---|
| 956d170 | pinned D-FINE commit | decisions/p3_licences.json |
| Apache-2.0 | D-FINE and RT-DETRv2 licence | decisions/p3_licences.json |
| ~540 TB / ~5,940 images | training split | docs/methodology.md §1.2 |
| ~60 TB | development split | docs/methodology.md §1.2 |
| 212 latent, 54 combined | whole TBX11K dataset | docs/methodology.md §1.1 table |
| 36 latent, 7 combined | official validation split (internal test = half of it) | docs/methodology.md §1.1 table, §1.2 |
| 512 × 512 | TBX11K image size | docs/methodology.md §7 |
| seed 0 | training seed (both sizes) | src/p3_det/configs/dfine_512.yaml, dfine_640.yaml |
| margin 0.10, CLAHE off | provisional preprocessing | src/p3_det/data/config.py |
| < 1 px | box/resize round-trip error (test) | tests/p3_det/test_det_data.py |
| 1e-3 on 3 fixtures | AP50 vs pycocotools agreement (test) | tests/p3_det/test_det_metrics.py |
| 5%, 0.001 grid | D18 rule (design) | decisions/p3_det_conf_threshold.json |
| 0.05 | detector contingency AP50 (design) | decisions/p3_det_input_size.json |
| 30 TB images | "small n" flag (design) | src/p3_det/metrics/detection.py |
| 20 images (10 TB/10 non-TB), 0.5 | D19 sanity check (design) | decisions/p3_gradcam_config.json |
| 5.5 GB | Grad-CAM VRAM limit (design) | decisions/p3_gradcam_config.json |
| 40 cases (20/20, 10 per cell) | review selection (design) | docs/p3_gradcam_review.md §1 |
| 7 scenario types, < 2 MB | PDF tests | tests/p3_ops/test_pdf_report.py |
| 50 parallel writers | logger concurrency test | tests/p3_ops/test_logger.py |
| < 2 s GPU, < 10 s CPU | E6 targets | README.md §1 |
| 100 images + 3 warm-up | E6 protocol (design) | src/p3_ops/latency.py |
| ≤ 500 ms | detector-only latency target | README.md P3-08 row |

## Still TBD (results that do not exist yet)

| Item | Waiting on |
|---|---|
| T4 batch size and peak memory per size | docs/p3_dfine_feasibility.md (P3-01 Kaggle run) |
| Exact training counts after deduplication | decisions/p1_splits.json (D01); runs' run_meta.json |
| COCO start-weights SHA-256 | artifacts/p3/det/runs/*/run_meta.json |
| Number of epochs | final config + decisions/p3_det_checkpoint.json (D17) |
| AP50 512 / 640 + CIs + n, chosen size, paired difference | decisions/p3_det_input_size.json (D16); artifacts/p3/det/select_size/d16_comparison.json |
| Chosen epoch, its AP50 + CI, checkpoint SHA | decisions/p3_det_checkpoint.json (D17) |
| Threshold, non-TB share (Wilson CI), TB sensitivity at threshold | decisions/p3_det_conf_threshold.json (D18, with guide_ack) |
| Internal-test AP50, image-level sensitivity, FROC @ 1/2/4 FP (+ CIs, n) | artifacts/p4/tables/t12_4_detection.csv (P4, after P3-13) |
| Subtype-stratified results | artifacts/p4/tables/t12_4_detection_subtype.csv (P4) |
| Contingency triggered or not | decisions/p3_det_input_size.json value.contingency |
| Grad-CAM VRAM measured | artifacts/p3/xai/vram_explain.json |
| Randomisation Spearman (signed and absolute), pass/fail | artifacts/p3/xai/sanity/randomisation_<model_id>.json |
| Development pointing accuracy / energy-in-box (+ CIs, n), both models | artifacts/p3/xai/loc_summary_development.json |
| Crop − full differences (paired CIs) | artifacts/p3/xai/loc_summary_development.json |
| Internal-test localisation (12.5) | artifacts/p3/xai/loc_<model_id>_internal_test_v1.parquet; P4 table t12_5 |
| Review session: date, attendees, comments, findings | docs/p3_gradcam_review.md (signed minutes) |
| PDF tests on P4's real fixtures and tbcore | P4-02 fixtures; tbcore |
| E6 GPU and CPU: median, p95, n, machine, verdict, per-stage, peak memory | artifacts/p3/latency/e6_gpu.csv, e6_cpu.csv, e6_*.md |
| Detector-only latency on the laptop | `make p3-latency-detect` output |
| Closure of the "all original boxes" gap | P1 uncropped box file (contract notes item 11) |
