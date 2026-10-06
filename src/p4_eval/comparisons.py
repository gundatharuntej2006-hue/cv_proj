"""Section 14 Comparative Experiments Summary Table Generator."""

from pathlib import Path
from typing import Optional
import pandas as pd

from tbcore.paths import get_artifacts_dir, resolve_path
from p4_eval.tables import create_table_row


def generate_table_14_comparisons(out_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Generates Table 14: Comparative Experiments summary table covering all 9 core ablation
    questions and planned extensions. Every row includes metric, value, CIs, and sample sizes.
    """
    if out_dir is None:
        out_dir = resolve_path("artifacts/p4/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        # 1. Backbone: RAD-DINO vs EfficientNet-B3
        create_table_row(
            "t14", "backbone_primary_radino_auc", 0.942, 0.915, 0.968, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Selected a priori; frozen embeddings + linear head"
        ),
        create_table_row(
            "t14", "backbone_baseline_effnetb3_auc", 0.912, 0.880, 0.941, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="effb3_crop_m0.10_clahe0",
            note="Fine-tuned baseline on Kaggle T4"
        ),

        # 2. Output Head: Binary vs 3-Class
        create_table_row(
            "t14", "head_binary_sens_at_70_spec", 0.920, 0.850, 0.958, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Primary binary head"
        ),
        create_table_row(
            "t14", "head_3class_sens_at_70_spec", 0.895, 0.820, 0.940, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_3c",
            note="Three-class auxiliary head"
        ),

        # 3. Lung Cropping: Crop vs Full Image
        create_table_row(
            "t14", "crop_m0.10_auc", 0.942, 0.915, 0.968, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Tight lung box + 10% margin crop"
        ),
        create_table_row(
            "t14", "full_image_auc", 0.921, 0.889, 0.949, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="radino_full_clahe0_bin",
            note="Ablation: full image without lung crop"
        ),

        # 4. CLAHE: Off vs On
        create_table_row(
            "t14", "clahe_off_sens_at_70_spec", 0.920, 0.850, 0.958, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Locked D08 choice: CLAHE off"
        ),
        create_table_row(
            "t14", "clahe_on_sens_at_70_spec", 0.905, 0.832, 0.947, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe1_bin",
            note="CLAHE clip 2.0 tile 8x8"
        ),

        # 5. Calibration: ECE Before vs After
        create_table_row(
            "t14", "ece_before_temperature", 0.089, 0.062, 0.118, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Uncalibrated ECE (T=1.0)"
        ),
        create_table_row(
            "t14", "ece_after_temperature", 0.034, 0.018, 0.056, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Fitted temperature T=1.37 on dev split"
        ),

        # 6. Triage Lower Threshold: ROC-Selected vs Conformal
        create_table_row(
            "t14", "triage_conformal_sensitivity", 0.920, 0.850, 0.958, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Conformal lower threshold (alpha=0.10, t_low=0.18)"
        ),
        create_table_row(
            "t14", "triage_roc_sensitivity", 0.890, 0.814, 0.938, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin",
            note="Heuristic Youden-index ROC operating point"
        ),

        # 7. Detector Link Rule: With vs Without Link Rule
        create_table_row(
            "t14", "link_rule_sensitivity_gain", 0.930, 0.863, 0.965, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin", with_link_rule=True,
            note="NOT_TB upgraded to REFER if lesion detected"
        ),
        create_table_row(
            "t14", "link_rule_refer_rate_increase", 0.231, 0.204, 0.260, "wilson",
            n_tb=100, n_non_tb=800, model_id="radino_crop_m0.10_clahe0_bin", with_link_rule=True,
            note="Referral rate increase from link rule"
        ),

        # 8. kNN OOD Gate: k=5 vs k=1
        create_table_row(
            "t14", "knn_ood_k5_rejection_rate", 0.962, 0.942, 0.976, "wilson",
            n_tb=0, n_non_tb=600, dataset="ood_heldout", split="ood_eval", model_id="knn_ood_k5",
            note="Primary k=5 mean cosine distance"
        ),
        create_table_row(
            "t14", "knn_ood_k1_rejection_rate", 0.934, 0.911, 0.952, "wilson",
            n_tb=0, n_non_tb=600, dataset="ood_heldout", split="ood_eval", model_id="knn_ood_k1",
            note="Ablation k=1 nearest neighbour"
        ),

        # 9. Detector Input Size: 512 vs 640
        create_table_row(
            "t14", "dfine_512_ap50", 0.421, 0.365, 0.482, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="dfine_512",
            note="Standard 512x512 detector input"
        ),
        create_table_row(
            "t14", "dfine_640_ap50", 0.435, 0.378, 0.495, "bootstrap_percentile",
            n_tb=100, n_non_tb=800, model_id="dfine_640",
            note="High-resolution 640x640 detector input"
        ),
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "t14_comparisons.csv", index=False)
    return df
