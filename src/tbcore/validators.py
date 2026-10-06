"""Validators for all project interface contracts (C01-C18)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import ValidationError

from tbcore.schemas import CaseResult, DecisionRecord


class ContractValidationError(ValueError):
    """Raised when a data contract file violates schema, dtype, or value constraints."""
    pass


# ---------------------------------------------------------------------------
# C01 — Split File Validator
# ---------------------------------------------------------------------------
def validate_c01_splits(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C01 file does not exist: {p}")

    df = pd.read_csv(p)
    required_cols = {
        "image_id", "rel_path", "official_split", "split", "category",
        "tb_label", "active_tb_label", "has_boxes", "n_boxes", "dup_of",
        "split_version", "seed"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C01 missing required columns: {missing}")

    if df["image_id"].duplicated().any():
        dups = df[df["image_id"].duplicated()]["image_id"].tolist()
        raise ContractValidationError(f"C01 contains duplicate image_ids: {dups[:5]}")

    valid_splits = {"train", "development", "calibration", "internal_test", "challenge_test", "excluded_duplicate"}
    invalid_splits = set(df["split"].unique()) - valid_splits
    if invalid_splits:
        raise ContractValidationError(f"C01 contains invalid split names: {invalid_splits}")

    return df


# ---------------------------------------------------------------------------
# C02 — External Manifest Validator
# ---------------------------------------------------------------------------
def validate_c02_external(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C02 file does not exist: {p}")

    df = pd.read_csv(p)
    required_cols = {"image_id", "rel_path", "source", "tb_label", "split", "orig_h", "orig_w", "manifest_version"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C02 missing required columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# C03 — Held-out OOD Manifest Validator
# ---------------------------------------------------------------------------
def validate_c03_ood(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C03 file does not exist: {p}")

    df = pd.read_csv(p)
    required_cols = {"image_id", "rel_path", "source", "ood_type", "role", "licence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C03 missing required columns: {missing}")

    valid_roles = {"ood_eval", "viewhead_train", "viewhead_dev", "viewhead_test"}
    invalid_roles = set(df["role"].unique()) - valid_roles
    if invalid_roles:
        raise ContractValidationError(f"C03 contains invalid roles: {invalid_roles}")

    return df


# ---------------------------------------------------------------------------
# C05 — Lung ROI Validator
# ---------------------------------------------------------------------------
def validate_c05_lung_roi(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C05 file does not exist: {p}")

    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    required_cols = {
        "image_id", "mask_path", "lung_area_frac", "n_components",
        "lung_x0", "lung_y0", "lung_x1", "lung_y1", "sanity_pass",
        "margin", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "fallback_full"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C05 missing required columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# C06 — Lesion Boxes COCO Validator
# ---------------------------------------------------------------------------
def validate_c06_coco(file_path: str | Path) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C06 file does not exist: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["images", "annotations", "categories"]:
        if key not in data:
            raise ContractValidationError(f"C06 missing COCO top-level key: {key}")

    return data


# ---------------------------------------------------------------------------
# C07 — RAD-DINO Embeddings Validator
# ---------------------------------------------------------------------------
def validate_c07_embeddings(file_path: str | Path) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C07 file does not exist: {p}")

    data = np.load(p)
    for key in ["image_id", "emb", "l2_normalised", "variant", "model_id"]:
        if key not in data:
            raise ContractValidationError(f"C07 missing npz key: {key}")

    emb = data["emb"]
    if len(emb.shape) != 2 or emb.shape[1] != 768:
        raise ContractValidationError(f"C07 embeddings must have shape [N, 768], got {emb.shape}")

    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# C08 — Classifier Scores Validator
# ---------------------------------------------------------------------------
def validate_c08_scores(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C08 file does not exist: {p}")

    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    required_cols = {"image_id", "dataset", "split", "model_id", "head", "p_tb_raw", "is_primary"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C08 missing required columns: {missing}")

    if not df["p_tb_raw"].between(0.0, 1.0).all():
        raise ContractValidationError("C08 p_tb_raw values must be in [0.0, 1.0]")

    return df


# ---------------------------------------------------------------------------
# C09 — Gate Scores Validator
# ---------------------------------------------------------------------------
def validate_c09_gate(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C09 file does not exist: {p}")

    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    required_cols = {"image_id", "quality_pass", "ood_score_k5", "ood_threshold_k5", "ood_pass", "gate_pass"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C09 missing required columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# C10 — Detections JSON Validator
# ---------------------------------------------------------------------------
def validate_c10_detections(file_path: str | Path) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C10 file does not exist: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["schema_version", "model_id", "input_size", "conf_threshold", "coords", "images"]:
        if key not in data:
            raise ContractValidationError(f"C10 missing key: {key}")

    if data["coords"] != "full512":
        raise ContractValidationError(f"C10 coords must be 'full512', got '{data['coords']}'")

    return data


# ---------------------------------------------------------------------------
# C12 — Triage Table Validator
# ---------------------------------------------------------------------------
def validate_c12_triage(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C12 file does not exist: {p}")

    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    required_cols = {
        "image_id", "split", "model_id", "p_tb_cal", "temperature",
        "t_lower", "t_upper", "category_pre_link", "link_fired",
        "category_final", "screen_positive", "gate_pass", "sanity_pass"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C12 missing required columns: {missing}")

    valid_cats = {"TB", "REFER", "NOT_TB"}
    invalid_cats = set(df["category_final"].unique()) - valid_cats
    if invalid_cats:
        raise ContractValidationError(f"C12 contains invalid triage categories: {invalid_cats}")

    return df


# ---------------------------------------------------------------------------
# C13 — Calibration Operating Curve Validator
# ---------------------------------------------------------------------------
def validate_c13_operating_curve(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C13 file does not exist: {p}")

    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    required_cols = {"t_lower", "sensitivity", "specificity", "refer_rate", "tb_rate", "n_tb", "n_non_tb", "is_locked_default"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C13 missing required columns: {missing}")

    if not df["is_locked_default"].any():
        raise ContractValidationError("C13 must contain at least one locked default operating point row")

    return df


# ---------------------------------------------------------------------------
# C14 — CaseResult JSON Validator
# ---------------------------------------------------------------------------
def validate_c14_case_result(file_or_data: str | Path | dict) -> CaseResult:
    if isinstance(file_or_data, (str, Path)):
        p = Path(file_or_data)
        if not p.exists():
            raise ContractValidationError(f"C14 file does not exist: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = file_or_data

    try:
        res = CaseResult(**data)
        return res
    except ValidationError as e:
        raise ContractValidationError(f"C14 CaseResult validation failed: {e}")


# ---------------------------------------------------------------------------
# C15 — Prediction Log JSONL Validator
# ---------------------------------------------------------------------------
def validate_c15_prediction_log(file_path: str | Path) -> List[dict]:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C15 file does not exist: {p}")

    records = []
    required_keys = {
        "ts", "run_id", "case_id", "status", "category_final",
        "link_fired", "gate_pass", "p_tb_cal", "t_lower", "t_upper",
        "operating_point", "model_version", "split_version",
        "decisions_hash", "git_sha", "latency_ms"
    }

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise ContractValidationError(f"C15 invalid JSON on line {line_num}: {e}")
            missing = required_keys - set(rec.keys())
            if missing:
                raise ContractValidationError(f"C15 missing keys on line {line_num}: {missing}")
            records.append(rec)

    return records


# ---------------------------------------------------------------------------
# C16 — Decision Record Validator
# ---------------------------------------------------------------------------
def validate_c16_decision(file_or_data: str | Path | dict) -> DecisionRecord:
    if isinstance(file_or_data, (str, Path)):
        p = Path(file_or_data)
        if not p.exists():
            raise ContractValidationError(f"C16 file does not exist: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = file_or_data

    try:
        rec = DecisionRecord(**data)
        return rec
    except ValidationError as e:
        raise ContractValidationError(f"C16 Decision validation failed: {e}")


# ---------------------------------------------------------------------------
# C17 — Near-Duplicate Candidates Validator
# ---------------------------------------------------------------------------
def validate_c17_near_duplicates(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C17 file does not exist: {p}")

    df = pd.read_csv(p)
    required_cols = {"image_id_a", "split_a", "image_id_b", "split_b", "cosine", "tau"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C17 missing required columns: {missing}")

    return df


# ---------------------------------------------------------------------------
# C18 — Evaluation Tables Validator
# ---------------------------------------------------------------------------
def validate_c18_tables(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise ContractValidationError(f"C18 file does not exist: {p}")

    df = pd.read_csv(p)
    required_cols = {
        "table_id", "metric", "value", "ci_low", "ci_high",
        "ci_method", "n_tb", "n_non_tb", "dataset", "split", "model_id"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ContractValidationError(f"C18 missing required columns: {missing}")

    # Mandatory n check: every row must have positive or non-null n_tb and n_non_tb
    if df["n_tb"].isna().any() or df["n_non_tb"].isna().any():
        raise ContractValidationError("C18 rows MUST carry non-null n_tb and n_non_tb counts (Section 12)")

    return df
