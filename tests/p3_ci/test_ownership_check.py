"""P3-00: tests for .github/scripts/ownership_check.py."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "ownership_check.py"
_spec = importlib.util.spec_from_file_location("ownership_check", SCRIPT)
oc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = oc  # dataclasses need the module registered
_spec.loader.exec_module(oc)

Change = oc.Change


@pytest.mark.parametrize("path, owner", [
    ("src/p1_data/api.py", "P1"),
    ("data/splits/tbx11k_splits_v1.csv", "P1"),
    ("data/splits/ood_heldout_v1.csv", "P2"),
    ("pyproject.toml", "P1"),
    ("src/p2_cls/api.py", "P2"),
    ("src/p3_det/data.py", "P3"),
    ("third_party/D-FINE", "P3"),
    (".github/workflows/ci.yml", "P3"),
    ("decisions/p3_det_input_size.json", "P3"),
    ("src/tbcore/seed.py", "P4"),
    ("contracts/schemas/c10_detections.json", "P4"),
    ("contracts/README.md", "P4"),
    ("README.md", "P4"),
    ("decisions/README.md", "P4"),
    (".github/CODEOWNERS", "P4"),
    ("tests/p2_cls/test_x.py", "P2"),
    ("docs/methodology.md", oc.FROZEN),
    ("docs/README.md", None),
    ("src/p3_det_extra.py", "P3"),  # "src/p3_*" matches any first segment starting p3_
    ("logs/predictions.jsonl", None),
])
def test_owner_of(path, owner):
    assert oc.owner_of(path) == owner


@pytest.mark.parametrize("branch, person", [
    ("p1/P1-05-dicom-ingest", "P1"), ("p3/P3-00-ci", "P3"),
    ("main", None), ("ccr/CCR-001", None), ("p5/x", None), ("xp2/y", None),
])
def test_branch_person(branch, person):
    assert oc.branch_person(branch) == person


def test_allowed_edit():
    changes = [Change("M", "src/p3_det/data.py"), Change("A", "tests/p3_det/test_data.py"),
               Change("M", ".github/workflows/ci.yml"), Change("A", "docs/README.md")]
    assert oc.check_ownership("p3/P3-04-adapter", changes) == []
    assert oc.check_data_guard(changes) == []


def test_forbidden_edit_p2_touches_p1():
    changes = [Change("M", "src/p2_cls/api.py"), Change("M", "src/p1_data/crop.py")]
    errors = oc.check_ownership("p2/P2-11-knn-gate", changes)
    assert len(errors) == 1
    assert "src/p1_data/crop.py" in errors[0] and "P1" in errors[0]


def test_forbidden_edit_contracts_and_frozen():
    changes = [Change("M", "contracts/case_result.schema.json"), Change("M", "docs/methodology.md")]
    errors = oc.check_ownership("p3/P3-03-mocks", changes)
    assert len(errors) == 2
    assert any("frozen" in e for e in errors)


def test_deleting_another_owners_file_is_forbidden():
    assert oc.check_ownership("p2/x", [Change("D", "src/p1_data/old.py")])


def test_non_person_branch_skips_ownership():
    assert oc.check_ownership("main", [Change("M", "src/p1_data/crop.py")]) == []


@pytest.mark.parametrize("path", [
    "data/raw/tbx11k/img_0001.png", "data/processed/v1/tbx11k/img512/a.JPG",
    "data/raw/shenzhen/b.jpeg", "data/raw/x/c.dcm", "data/splits/sneaky.png",
])
def test_staged_image_under_data_fails(path):
    errors = oc.check_data_guard([Change("A", path)])
    assert len(errors) == 1 and path in errors[0]


def test_data_guard_allows_split_csv_deletions_and_images_elsewhere():
    changes = [Change("A", "data/splits/tbx11k_splits_v1.csv"),
               Change("D", "data/raw/tbx11k/img_0001.png"),  # removing a leaked image is fine
               Change("A", "artifacts/p3/xai/overlay_example.png")]
    assert oc.check_data_guard(changes) == []


def test_data_guard_applies_to_owner_branch_too():
    # P1 owns data/, but images are still refused.
    assert oc.check_data_guard([Change("A", "data/raw/tbx11k/a.png")])


def test_parse_name_status():
    out = "M\tsrc/p3_det/a.py\nA\tdata/x.png\nD\told.txt\n"
    assert oc.parse_name_status(out) == [
        Change("M", "src/p3_det/a.py"), Change("A", "data/x.png"), Change("D", "old.txt")]


# --- end-to-end against a real temporary git repo (what CI runs) ------------

def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "ci@example.invalid")
    _git(tmp_path, "config", "user.name", "ci")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("x\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commit_on_branch(repo, branch, files):
    _git(repo, "switch", "-q", "-c", branch)
    for rel in files:
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"\x89PNG" if rel.endswith(".png") else b"x\n")
    _git(repo, "add", "-f", ".")
    _git(repo, "commit", "-q", "-m", "change")


def test_cli_bad_p2_pr_touching_p1_fails(repo, capsys):
    _commit_on_branch(repo, "p2/P2-99-bad", ["src/p2_cls/a.py", "src/p1_data/b.py"])
    assert oc.main(["--base", "main", "--head", "HEAD"]) == 1
    assert "src/p1_data/b.py" in capsys.readouterr().out


def test_cli_pr_staging_data_image_fails(repo, capsys):
    _commit_on_branch(repo, "p1/P1-99-bad", ["data/raw/tbx11k/img_0001.png"])
    assert oc.main(["--base", "main", "--head", "HEAD"]) == 1
    assert "data guard" in capsys.readouterr().out


def test_cli_good_p3_pr_passes(repo):
    _commit_on_branch(repo, "p3/P3-00-ci", ["src/p3_ops/x.py", ".github/workflows/ci.yml"])
    assert oc.main(["--base", "main", "--head", "HEAD"]) == 0


def test_cli_staged_mode(repo):
    _git(repo, "switch", "-q", "-c", "p2/P2-98-staged")
    (repo / "src" / "p1_data").mkdir(parents=True)
    (repo / "src" / "p1_data" / "c.py").write_text("x\n")
    _git(repo, "add", ".")
    assert oc.main(["--staged"]) == 1


def test_cli_all_tracked_mode(repo):
    _commit_on_branch(repo, "p1/leak", ["data/raw/a.png"])
    assert oc.main(["--all-tracked"]) == 1
