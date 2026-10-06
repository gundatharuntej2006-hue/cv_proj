"""R4 Runtime Python API: Pipeline execution and triage entry points."""

from pathlib import Path
from typing import List, Union

from tbcore.schemas import CaseResult
from p4_pipeline.orchestrator import TriagePipelineOrchestrator
from p4_stats.triage import triage_case

# Global default orchestrator instance
_default_orchestrator = None


def get_default_orchestrator() -> TriagePipelineOrchestrator:
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = TriagePipelineOrchestrator()
    return _default_orchestrator


def run_case(path: Union[str, Path]) -> CaseResult:
    """
    R4 entry point: Executes end-to-end CAD triage pipeline on a single radiograph.
    Never raises an uncaught exception; returns CaseResult with UNREADABLE or REJECTED_GATE if errors occur.
    """
    orch = get_default_orchestrator()
    return orch.process_file(path)


def run_batch(paths: List[Union[str, Path]]) -> List[CaseResult]:
    """
    R4 entry point: Batch processes radiographs with strict per-file error isolation.
    """
    orch = get_default_orchestrator()
    return orch.process_batch(paths)


def triage(p_tb_cal: float, n_lesions: int, t_lower: float, t_upper: float) -> str:
    """
    R4 entry point: Computes triage category ("TB", "REFER", "NOT_TB") using conformal thresholds and link rule.
    """
    res = triage_case(p_tb_cal, n_lesions, t_lower, t_upper)
    return str(res["category_final"])
