# -*- coding: utf-8 -*-
"""P2's machine-readable reading of the five contracts P2 produces: C03, C07, C08, C09, C17.

Authority note. `contracts/specs/*.md` and Person 4's `tbcore` validators are the
authority. This file exists because the mock generator (P2-02) and P2's own tests need
the same field list in code, and P2 must not edit `tbcore`. When `tbcore` lands,
`tests/p2_cls/test_contracts_agree.py` checks this file against it, never the reverse.
Any change here without a CCR is a bug.
"""
import pyarrow as pa

CAT = pa.dictionary(pa.int32(), pa.string())          # pandas reads this back as category

EMB_DIM = 768            # measured in P2-01, not assumed: RAD-DINO hidden_size
SCHEMA_VERSION = "v1"
TAU_DEFAULT = 0.98       # D06 proposed value for the near-duplicate cosine threshold

# --- C03: held-out OOD collection (CSV, committed) --------------------------------
C03_COLUMNS = ["image_id", "rel_path", "source", "ood_type", "role", "licence"]
C03_OOD_TYPES = ["lateral_cxr", "other_body_part", "non_medical"]
C03_ROLES = ["ood_eval", "viewhead_train", "viewhead_dev", "viewhead_test"]

# --- C07: RAD-DINO embeddings (NPZ) -----------------------------------------------
C07_KEYS = ["image_id", "emb", "l2_normalised", "variant", "model_id", "preproc_version"]

# --- C08: classifier scores (Parquet) ---------------------------------------------
C08_HEADS = ["binary", "three_class"]
C08_SCHEMA = pa.schema([
    pa.field("image_id", pa.string(), nullable=False),
    pa.field("dataset", CAT, nullable=False),
    pa.field("split", CAT, nullable=False),
    pa.field("model_id", pa.string(), nullable=False),
    pa.field("head", CAT, nullable=False),
    pa.field("variant", pa.string(), nullable=False),
    pa.field("clahe", pa.bool_(), nullable=False),
    pa.field("margin", pa.float32()),                       # null for full-image variants
    pa.field("logit_tb", pa.float32()),                     # null for the three-class head
    # A variable-length list, not pa.list_(pa.float32(), 3): Parquet has no fixed-size
    # list type, so a null in a fixed-size list column comes back as an empty list and the
    # file will not even re-read. The length-3 invariant is enforced by the validator below.
    pa.field("logits_3c", pa.list_(pa.float32())),          # null for the binary head
    pa.field("p_tb_raw", pa.float32(), nullable=False),
    pa.field("p3_healthy", pa.float32()),
    pa.field("p3_sick_non_tb", pa.float32()),
    pa.field("p3_tb", pa.float32()),
    pa.field("is_primary", pa.bool_(), nullable=False),
    pa.field("model_sha", pa.string(), nullable=False),
    pa.field("seed", pa.int32(), nullable=False),
])

# --- C09: gate scores (Parquet) ---------------------------------------------------
C09_QUALITY_REASONS = ["blank", "under_exposed", "over_exposed", "heavily_cropped"]
C09_SCHEMA = pa.schema([
    pa.field("image_id", pa.string(), nullable=False),
    pa.field("dataset", CAT, nullable=False),
    pa.field("split_or_role", pa.string(), nullable=False),
    pa.field("quality_pass", pa.bool_(), nullable=False),
    pa.field("quality_reasons", pa.string(), nullable=False),   # "" when it passed
    pa.field("ood_score_k5", pa.float32(), nullable=False),
    pa.field("ood_score_k1", pa.float32(), nullable=False),
    pa.field("ood_threshold_k5", pa.float32(), nullable=False),
    pa.field("ood_pass", pa.bool_(), nullable=False),
    pa.field("gate_pass", pa.bool_(), nullable=False),
])

# --- C17: near-duplicate candidates (CSV) -----------------------------------------
C17_COLUMNS = ["image_id_a", "split_a", "image_id_b", "split_b", "cosine", "tau"]

# --- the D14 model registry the mocks pretend to contain --------------------------
# Five entries, exactly the set P2-13 must ship: primary, alternate head, full-image
# variant, CLAHE alternate, EfficientNet-B3 baseline.
REGISTRY = [
    {"model_id": "radino_crop_m0.10_clahe0_bin", "head": "binary",
     "variant": "crop_m0.10_clahe0", "clahe": False, "margin": 0.10, "is_primary": True},
    {"model_id": "radino_crop_m0.10_clahe0_3c", "head": "three_class",
     "variant": "crop_m0.10_clahe0", "clahe": False, "margin": 0.10, "is_primary": False},
    {"model_id": "radino_full_clahe0_bin", "head": "binary",
     "variant": "full_clahe0", "clahe": False, "margin": None, "is_primary": False},
    {"model_id": "radino_crop_m0.10_clahe1_bin", "head": "binary",
     "variant": "crop_m0.10_clahe1", "clahe": True, "margin": 0.10, "is_primary": False},
    {"model_id": "effb3_crop_m0.10_clahe0", "head": "binary",
     "variant": "crop_m0.10_clahe0", "clahe": False, "margin": 0.10, "is_primary": False},
]


def check_c08_invariants(df):
    """Invariants Parquet itself cannot express. Returns a list of human-readable problems.
    Kept here so the mock generator and the real P2-07/P2-13 writers are checked the same
    way, and so Person 4's tbcore validator has something concrete to agree with."""
    bad = []
    three = df[df["head"] == "three_class"]
    binary = df[df["head"] == "binary"]
    if len(three) and not three["logits_3c"].map(lambda v: v is not None and len(v) == 3).all():
        bad.append("three_class rows must carry exactly 3 logits in logits_3c")
    if len(three) and three["logit_tb"].notna().any():
        bad.append("three_class rows must leave logit_tb null")
    if len(binary) and binary["logits_3c"].notna().any():
        bad.append("binary rows must leave logits_3c null")
    if len(binary) and binary["logit_tb"].isna().any():
        bad.append("binary rows must carry logit_tb")
    if len(three):
        s3 = three[["p3_healthy", "p3_sick_non_tb", "p3_tb"]].sum(axis=1)
        if not ((s3 - 1.0).abs() < 1e-3).all():
            bad.append("three-class probabilities must sum to 1")
        if not ((three["p3_tb"] - three["p_tb_raw"]).abs() < 1e-6).all():
            bad.append("for a three-class row, p_tb_raw must equal p3_tb")
    if df.groupby(["image_id", "model_id"]).size().max() > 1:
        bad.append("one row per (image_id, model_id) only")
    if int(df["is_primary"].groupby(df["model_id"]).max().sum()) != 1:
        bad.append("exactly one model_id may be is_primary")
    if not df["p_tb_raw"].between(0, 1).all():
        bad.append("p_tb_raw must lie in [0, 1]")
    return bad
