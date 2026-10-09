"""P3-14: prediction logger / R3 log_case (C15).

Input: the 7 CaseResult scenario types of P4-02, as test-only stand-ins (tests/p3_mock/p3_case_results.py)
until P4's fixtures exist. Validation: the C15 stand-in validator (tests/p3_mock/standin_contracts.py).
"""
from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")
REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "tests" / "p3_mock"):
    sys.path.insert(0, str(p))

import standin_contracts as sc  # noqa: E402
from p3_case_results import ANALYSED, P4_SCENARIOS  # noqa: E402

from p3_ops import api, logger  # noqa: E402


@pytest.fixture
def log(tmp_path, monkeypatch):
    p = tmp_path / "logs" / "predictions.jsonl"            # folder does not exist yet: created on first write
    monkeypatch.setenv("TB_LOG_PATH", str(p))
    monkeypatch.delenv("TB_RUN_ID", raising=False)
    monkeypatch.delenv("TB_MOCK", raising=False)
    return p


def lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="ascii").splitlines()]


# --- one line per call, C15 ------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(P4_SCENARIOS))
def test_one_valid_c15_line_per_call(log, name):
    cr = P4_SCENARIOS[name]
    api.log_case(cr, "locked_default")
    out = sc.validate_c15_file(log)
    assert len(out) == 1
    line = out[0]
    assert list(line) == list(logger.FIELDS)
    assert line["case_id"] == cr["case_id"] and line["status"] == cr["status"]
    assert all(line[k] == cr["versions"][k] for k in logger.VERSION_FIELDS)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", line["ts"])
    if cr["status"] == "ANALYSED":
        assert line["p_tb_cal"] == cr["classifier"]["p_tb_cal"]
        assert line["category_final"] == cr["triage"]["category_final"]
        assert line["link_fired"] == cr["triage"]["link_fired"]
        assert line["latency_ms"] == cr["timing_ms"]["total"] and line["run_id"] == "r0012"


def test_rejected_and_unreadable_write_nulls_but_keep_versions(log):
    api.log_case(P4_SCENARIOS["rejected_gate"], "locked_default")
    api.log_case(P4_SCENARIOS["unreadable"], "locked_default")
    rej, unr = lines(log)
    for line in (rej, unr):
        assert line["p_tb_cal"] is None and line["category_final"] is None and line["link_fired"] is None
        assert line["t_lower"] is None and line["t_upper"] is None
        assert all(isinstance(line[k], str) and line[k] for k in logger.VERSION_FIELDS)
    assert rej["gate_pass"] is False and unr["gate_pass"] is None
    assert unr["latency_ms"] is None


def test_each_call_appends_exactly_one_line(log):
    for k, cr in enumerate(P4_SCENARIOS.values(), start=1):
        api.log_case(cr, "locked_default")
        assert len(log.read_text().splitlines()) == k
    assert log.read_text().endswith("\n")


def test_user_override_written_exactly(log):
    api.log_case(ANALYSED, "user_override")
    assert lines(log)[0]["operating_point"] == "user_override"
    for bad in ("User_Override", "override", "", None):
        with pytest.raises(logger.LogError):
            api.log_case(ANALYSED, bad)
    assert len(lines(log)) == 1                              # refused calls write nothing


# --- required fields -------------------------------------------------------------------------------

@pytest.mark.parametrize("field", logger.VERSION_FIELDS)
@pytest.mark.parametrize("how", ["missing", "none", "empty"])
def test_missing_version_field_raises_and_writes_nothing(log, field, how):
    cr = copy.deepcopy(ANALYSED)
    if how == "missing":
        del cr["versions"][field]
    else:
        cr["versions"][field] = None if how == "none" else ""
    with pytest.raises(logger.LogError, match=field):
        api.log_case(cr, "locked_default")
    assert not log.exists() or log.read_text() == ""


def test_missing_versions_block_raises(log):
    cr = copy.deepcopy(ANALYSED)
    cr["versions"] = None
    with pytest.raises(logger.LogError, match="model_version"):
        api.log_case(cr, "locked_default")


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(case_id="John_Doe/1985"), lambda c: c.update(case_id="../etc/passwd"),
    lambda c: c.update(status="DONE"), lambda c: c["triage"].update(category_final="maybe"),
    lambda c: c["classifier"].update(p_tb_cal="0.5"),
    lambda c: c["versions"].update(git_sha=r"C:\Users\jane.doe\repo"),
    lambda c: c["versions"].update(model_version="reg v1 (Jane)"),
])
def test_unsafe_or_invalid_values_refused(log, mutate):
    cr = copy.deepcopy(ANALYSED)
    mutate(cr)
    with pytest.raises(logger.LogError):
        api.log_case(cr, "locked_default")


# --- no PHI: grep test ------------------------------------------------------------------------------

IDENTIFIER_PATTERNS = {
    "DICOM tag number": r"\(\s*[0-9A-Fa-f]{4}\s*,\s*[0-9A-Fa-f]{4}\s*\)",
    "DICOM keyword": r"Patient(Name|ID|BirthDate|Sex|Age|Address)|StudyDate|AccessionNumber|InstitutionName",
    "DICOM person name": r"[A-Za-z]+\^[A-Za-z]+",
    "Windows user path": r"[A-Za-z]:\\\\|\\\\Users\\\\|C:/Users/",
    "Unix user path": r"/home/|/Users/|/root/",
    "image file name": r"\.(dcm|dicom|png|jpe?g|tiff?)\b",
    "personal name": r"\b(Jane|John|Doe|Smith|Kumar|Sharma)\b",
    "date of birth": r"(?<!\d)(19|20)\d\d-\d\d-\d\d(?![\dT])",     # not the log's own "...T" timestamps
}


def leaky(cr: dict) -> dict:
    """The same CaseResult with identifiers planted in every free-text place a CaseResult has."""
    c = copy.deepcopy(cr)
    if c.get("input"):
        c["input"]["error"] = "PatientName=DOE^JANE (0010,0010) file C:\\Users\\jane.doe\\scan_1985-04-02.dcm"
    if c.get("gate"):
        c["gate"]["quality_reasons"] = ["PatientID 12345 /home/jsmith/x.png"]
    if c.get("segmentation"):
        c["segmentation"]["fail_reasons"] = ["John Smith"]
    if c.get("explainability"):
        c["explainability"] = {"heatmap_path": "runs/r0012/C:\\Users\\Kumar\\cam.npy",
                               "overlay_path": "/Users/sharma/runs/r0012/overlay.png"}
    c["notes"] = "InstitutionName=General Hospital"
    return c


def test_grep_log_built_from_fixtures_has_no_identifiers(log):
    for cr in P4_SCENARIOS.values():
        for op in logger.OPERATING_POINTS:
            api.log_case(cr, op)
            api.log_case(leaky(cr), op)
    text = log.read_text(encoding="ascii")
    assert len(text.splitlines()) == 4 * len(P4_SCENARIOS)
    hits = {name: re.findall(pat, text) for name, pat in IDENTIFIER_PATTERNS.items()}
    assert not any(hits.values()), {k: v for k, v in hits.items() if v}
    sc.validate_c15_file(log)
    assert {l["case_id"] for l in lines(log)} == {cr["case_id"] for cr in P4_SCENARIOS.values()}


def test_grep_patterns_do_catch_identifiers():
    sample = json.dumps(leaky(ANALYSED))
    assert all(re.search(p, sample) for name, p in IDENTIFIER_PATTERNS.items() if name != "personal name") \
        and re.search(IDENTIFIER_PATTERNS["personal name"], sample)


# --- concurrency -----------------------------------------------------------------------------------

def test_50_parallel_appends_threads(log):
    def work(k):
        cr = copy.deepcopy(ANALYSED)
        cr["case_id"] = f"c_{k:08x}"
        api.log_case(cr, "locked_default")
    ts = [threading.Thread(target=work, args=(k,)) for k in range(50)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    out = sc.validate_c15_file(log)
    assert len(out) == 50 and len({l["case_id"] for l in out}) == 50


WORKER = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from p3_case_results import ANALYSED
from p3_ops import logger
import copy
for i in range(int(sys.argv[4])):
    cr = copy.deepcopy(ANALYSED)
    cr["case_id"] = f"c_p{sys.argv[3]}_{i:03d}"
    logger.log_case(cr, "locked_default")
"""


def test_50_parallel_appends_processes(log):
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, str(REPO / "src"), str(REPO / "tests" / "p3_mock"),
                               str(p), "10"]) for p in range(5)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    out = sc.validate_c15_file(log)
    assert len(out) == 50 and len({l["case_id"] for l in out}) == 50


def test_partial_line_from_a_crash_does_not_corrupt_the_next_line(log):
    log.parent.mkdir(parents=True)
    log.write_bytes(b'{"ts":"2026-11-30T10:00:00.000+00:00","run_id":"r1","case_')   # writer died mid-line
    api.log_case(ANALYSED, "locked_default")
    raw = log.read_text().splitlines()
    assert len(raw) == 2 and json.loads(raw[1])["case_id"] == ANALYSED["case_id"]


def test_default_path_and_folder_creation(tmp_path, monkeypatch):
    monkeypatch.delenv("TB_LOG_PATH", raising=False)
    monkeypatch.setenv("TB_MOCK", "1")
    assert logger.log_path() == REPO / "mock" / "logs" / "predictions.jsonl"
    monkeypatch.delenv("TB_MOCK")
    assert logger.log_path() == REPO / "logs" / "predictions.jsonl"
