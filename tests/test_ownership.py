from pathlib import Path


def test_codeowners_file_exists():
    root_co = Path("CODEOWNERS")
    gh_co = Path(".github/CODEOWNERS")
    assert root_co.exists() or gh_co.exists()

    content = (root_co if root_co.exists() else gh_co).read_text(encoding="utf-8")
    assert "p1_data" in content
    assert "p2_cls" in content
    assert "p3_det" in content
    assert "p4_stats" in content
    assert "tbcore" in content
    assert "contracts" in content


def test_module_structure_matches_plan():
    required_dirs = [
        "src/p1_data", "src/p1_app", "src/p1_viewhead",
        "src/p2_cls", "src/p2_gate",
        "src/p3_det", "src/p3_xai", "src/p3_ops",
        "src/tbcore",
        "src/p4_stats", "src/p4_pipeline", "src/p4_eval",
        "contracts", "decisions", "data", "tests",
        "notebooks", "artifacts", "mock", "runs",
        "logs", "report", "docs", "mk", ".github"
    ]
    for d in required_dirs:
        p = Path(d)
        assert p.exists() and p.is_dir(), f"Required directory {d} missing!"
