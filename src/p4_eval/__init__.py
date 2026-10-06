"""p4_eval - No-test-tuning gate, evaluation tables, challenge submission, and comparisons."""

from p4_eval.challenge import format_challenge_submission, generate_table_12_8_challenge, log_challenge_submission
from p4_eval.comparisons import generate_table_14_comparisons
from p4_eval.external import evaluate_external_cohort, run_all_external_evaluations
from p4_eval.gate import check_all_decisions, issue_token, verify_gate_prerequisites
from p4_eval.recalibration import run_external_recalibration
from p4_eval.tables import (
    create_table_row,
    export_evaluation_tables,
    generate_table_12_1_classification,
    generate_table_12_2_calibration,
    generate_table_12_3_triage,
    generate_table_12_6_gate,
    generate_table_12_7_segmentation_sanity,
)

__all__ = [
    "check_all_decisions",
    "issue_token",
    "verify_gate_prerequisites",
    "create_table_row",
    "export_evaluation_tables",
    "generate_table_12_1_classification",
    "generate_table_12_2_calibration",
    "generate_table_12_3_triage",
    "generate_table_12_6_gate",
    "generate_table_12_7_segmentation_sanity",
    "format_challenge_submission",
    "log_challenge_submission",
    "generate_table_12_8_challenge",
    "evaluate_external_cohort",
    "run_all_external_evaluations",
    "generate_table_14_comparisons",
    "run_external_recalibration",
]
