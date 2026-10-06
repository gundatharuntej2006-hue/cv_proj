# Contract Specifications (C02–C13, C15, C17)

### C02 — External Manifest
- **File:** `data/splits/external_v1.csv`
- **Producer:** P1 | **Consumers:** P2, P4
- **Fields:** `image_id` (str), `rel_path` (str), `source` (str), `tb_label` (int 0/1), `split` (str: external_test), `orig_h` (int), `orig_w` (int), `manifest_version` (str)

### C03 — Held-Out OOD Collection Manifest
- **File:** `data/splits/ood_heldout_v1.csv`
- **Producer:** P2 | **Consumers:** P4, P1
- **Fields:** `image_id` (str), `rel_path` (str), `source` (str), `ood_type` (category), `role` (str: ood_eval, viewhead_train, viewhead_dev, viewhead_test), `licence` (str)

### C04 — Preprocessed Image Store
- **Files:** `data/processed/v1/{dataset}/img512/{image_id}.png`, `.../img512_clahe/{image_id}.png`, manifest `data/processed/v1/manifest.parquet`
- **Producer:** P1 | **Consumers:** P2, P3
- **Properties:** 512x512 uint8 single-channel PNG; CLAHE clip 2.0, tile 8x8.

### C05 — Lung ROI Table
- **File:** `data/processed/v1/lung_roi.parquet`
- **Producer:** P1 | **Consumers:** P2, P3, P4
- **Fields:** `image_id`, `mask_path`, `lung_area_frac`, `n_components`, `lung_x0..y1`, `sanity_pass`, `margin`, `crop_x0..y1`, `fallback_full`

### C06 — Lesion Boxes (COCO)
- **Files:** `data/boxes/v1/coco_{split}_m{margin}.json` + `outside_crop_v1.csv`
- **Producer:** P1 | **Consumers:** P3, P4
- **Format:** COCO JSON object with `images`, `annotations` (category_id=1 `tb_lesion`), `categories`.

### C07 — RAD-DINO Embeddings
- **Files:** `artifacts/p2/emb/radino_{dataset}_{variant}_v1.npz`
- **Producer:** P2 | **Consumers:** P2, P1
- **Keys:** `image_id` [N], `emb` [N, 768] (float16, L2-normalized), `l2_normalised` (bool), `variant` (str), `model_id` (str)

### C08 — Classifier Scores
- **Files:** `artifacts/p2/scores/scores_{dataset}_{split}_v1.parquet`
- **Producer:** P2 | **Consumers:** P4
- **Fields:** `image_id`, `dataset`, `split`, `model_id`, `head`, `variant`, `clahe`, `margin`, `logit_tb`, `logits_3c`, `p_tb_raw`, `p3_healthy`, `p3_sick_non_tb`, `p3_tb`, `is_primary`, `model_sha`, `seed`

### C09 — Gate Scores
- **Files:** `artifacts/p2/gate/gate_{dataset}_{split}_v1.parquet`
- **Producer:** P2 | **Consumers:** P4
- **Fields:** `image_id`, `dataset`, `split_or_role`, `quality_pass`, `quality_reasons`, `ood_score_k5`, `ood_score_k1`, `ood_threshold_k5`, `ood_pass`, `gate_pass`

### C10 — Detections
- **Files:** `artifacts/p3/det/detections_{dataset}_{split}_v1.json`
- **Producer:** P3 | **Consumers:** P4
- **Format:** JSON object with `schema_version`, `model_id`, `input_size`, `conf_threshold`, `coords`="full512", `images` list.

### C11 — Grad-CAM Heatmaps & Localisation Scores
- **Files:** `artifacts/p3/xai/{model_id}/{split}/{image_id}.npy` + `loc_{model_id}_{split}_v1.parquet`
- **Producer:** P3 | **Consumers:** P4, P1
- **Format:** float16 [512,512] numpy array in [0,1] range + evaluation metrics parquet.

### C12 — Triage Table
- **Files:** `artifacts/p4/triage/triage_{dataset}_{split}_v1.parquet`
- **Producer:** P4 | **Consumers:** P1, P2, P3
- **Fields:** `image_id`, `split`, `model_id`, `p_tb_cal`, `temperature`, `t_lower`, `t_upper`, `category_pre_link`, `link_fired`, `category_final`, `screen_positive`, `gate_pass`, `sanity_pass`, `n_lesions`

### C13 — Calibration Operating Curve
- **Files:** `artifacts/p4/triage/operating_curve_calibration_v1.parquet`
- **Producer:** P4 | **Consumers:** P1
- **Fields:** `t_lower`, `sensitivity`, `specificity`, `refer_rate`, `tb_rate`, `n_tb`, `n_non_tb`, `is_locked_default`

### C15 — Prediction Log
- **File:** `logs/predictions.jsonl`
- **Producer:** P3 | **Consumers:** P4, P1
- **Format:** JSON Lines, append-only, zero PHI.

### C17 — Near-Duplicate Candidates
- **File:** `artifacts/p2/dedup/near_dup_candidates_v1.csv`
- **Producer:** P2 | **Consumers:** P1
- **Fields:** `image_id_a`, `split_a`, `image_id_b`, `split_b`, `cosine`, `tau`
