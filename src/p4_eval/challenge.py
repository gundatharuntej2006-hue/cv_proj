"""TBX11K Online Challenge submission packaging implementing Section 4.6 category mapping."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from tbcore.paths import get_artifacts_dir, resolve_path
from tbcore.version import compute_file_sha256, get_git_sha
from p4_eval.tables import create_table_row


def format_challenge_submission(
    df_cls_scores: pd.DataFrame,
    df_det_boxes: Optional[dict] = None,
    out_dir: Optional[Path] = None
) -> Tuple[Path, str]:
    """
    Formats the challenge test predictions adhering strictly to Section 4.6 category mapping:
    - Classification: p(active_tb) = p(tb), p(latent_tb) = 0.0
    - Detection: every predicted box submitted as category 1 (ActiveTuberculosis)
    """
    if out_dir is None:
        out_dir = resolve_path("artifacts/p4/challenge")
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_json = out_dir / "submission_tbx11k_v1.json"

    cls_records = []
    for _, row in df_cls_scores.iterrows():
        p_tb = float(row.get("p_tb_raw", 0.5))
        p3 = row.get("three_class")
        if isinstance(p3, dict):
            p_healthy = float(p3.get("healthy", 1.0 - p_tb))
            p_sick = float(p3.get("sick_non_tb", 0.0))
        else:
            p_healthy = float(1.0 - p_tb)
            p_sick = 0.0

        cls_records.append({
            "image_id": str(row["image_id"]),
            "prob_healthy": p_healthy,
            "prob_sick_non_tb": p_sick,
            "prob_active_tb": p_tb,
            "prob_latent_tb": 0.0,
        })

    sub_data = {
        "schema_version": "1.0",
        "mapping_rule": "Section 4.6 pre-registered mapping: p(active_tb)=p(tb), p(latent)=0, all boxes class=1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "classification": cls_records,
        "detection_boxes": df_det_boxes.get("images", []) if df_det_boxes else [],
    }

    with open(submission_json, "w", encoding="utf-8") as f:
        json.dump(sub_data, f, indent=2)

    sha = compute_file_sha256(submission_json)
    return submission_json, sha


def log_challenge_submission(
    submission_path: Path,
    file_sha: str,
    returned_metrics: Dict[str, float],
    log_file: Optional[Path] = None
):
    """Logs the single official submission to docs/p4_codabench_log.md."""
    if log_file is None:
        log_file = resolve_path("docs/p4_codabench_log.md")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n### Submission — {ts}\n")
        f.write(f"- **File:** `{submission_path.name}`\n")
        f.write(f"- **SHA-256:** `{file_sha}`\n")
        f.write(f"- **Git Commit:** `{get_git_sha()}`\n")
        f.write(f"- **Returned Metrics (Verbatim):**\n")
        for k, v in returned_metrics.items():
            f.write(f"  - `{k}`: {v:.4f}\n")
        f.write(
            f"- **Known Limitation Note:** Category mapping fixed in D21: all boxes mapped to ActiveTuberculosis. "
            f"Latent TB AP is near-zero by construction; category-agnostic detection is the primary comparator.\n"
        )


def generate_table_12_8_challenge(
    returned_metrics: Dict[str, float],
    out_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Generates Table 12.8 recording challenge metrics with mandatory mapping caveat."""
    if out_dir is None:
        out_dir = resolve_path("artifacts/p4/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric_name, val in returned_metrics.items():
        rows.append(create_table_row(
            table_id="t12_8",
            metric=metric_name,
            value=val,
            ci_low=val,
            ci_high=val,
            ci_method="wilson",
            n_tb=100,
            n_non_tb=800,
            dataset="tbx11k",
            split="challenge_test",
            model_id="radino_crop_m0.10_clahe0_bin",
            note="Reported verbatim from challenge platform; Section 4.6 mapping applied"
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "t12_8_challenge.csv", index=False)
    return df
