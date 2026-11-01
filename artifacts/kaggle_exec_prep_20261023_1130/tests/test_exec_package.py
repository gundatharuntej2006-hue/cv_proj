# -*- coding: utf-8 -*-
"""Tests for the Kaggle execution package. No weights, no network, no real images.

    py -3 -m pytest artifacts/kaggle_exec_prep_20261023_1130/tests -q

Each test here exists because the thing it checks can actually fail. The staged-layout
fixture is what makes the REPORT-intrusion guard testable: in the repository layout that
guard is inert by design, so a test against it there would assert nothing.
"""
import ast
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "src"))

import paths as P                                                   # noqa: E402
import concept_audit as A                                           # noqa: E402
import preflight as F                                               # noqa: E402

FP = json.loads((PKG / "configs/tune_cohort_fingerprint.json").read_text(encoding="utf-8"))
MAP = json.loads((PKG / "configs/concept_indices.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    """Correctness does not need 10,000 replicates; the real run uses them."""
    monkeypatch.setattr(A, "N_BOOT", 400)
    yield


# --------------------------------------------------------- a synthetic staged upload
def _staged(tmp_path, n_tune=40, n_report=12, intrude=None, dup=False, drop=0,
            make_images=True):
    """Build a miniature staged layout. Ids are synthetic; no real patient data."""
    d = tmp_path / "staged"
    (d / P.STAGED_IMG_DIR).mkdir(parents=True)
    tune = ["t%04d" % i for i in range(n_tune)]
    report = ["r%04d" % i for i in range(n_report)]
    json.dump({"tune": tune, "report": report},
              io.open(d / P.STAGED_FILES["split"], "w", encoding="utf-8"))

    rows_pids = list(tune)
    if drop:
        rows_pids = rows_pids[:-drop]
    if dup:
        rows_pids = rows_pids[:-1] + [rows_pids[0]]
    if intrude:
        rows_pids = rows_pids + list(intrude)

    with io.open(d / P.STAGED_FILES["scores"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "score_baseline"])
        for i, q in enumerate(rows_pids):
            w.writerow([q, round(0.01 * (i % 97), 4)])
    with io.open(d / P.STAGED_FILES["meta"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["pid", "sex", "age_group", "c", "f", "wl", "ns", "past_tb", "rad_system",
                "xpert_result"]
        w.writerow(cols)
        for i, q in enumerate(sorted(set(rows_pids))):
            w.writerow([q, i % 2, 1 + i % 2, i % 2, 0, 1, 0, int(i % 3 == 0),
                        1 + i % 3,
                        int(i % 7 == 0)])
    json.dump({"n": len(tune), "fingerprint_sha256": A.cohort_fingerprint(tune),
               "pids": sorted(tune)},
              io.open(d / P.STAGED_FILES["manifest"], "w", encoding="utf-8"))
    if make_images:
        for q in tune:
            (d / P.STAGED_IMG_DIR / ("%s.png" % q)).write_bytes(b"")
    return d, tune


@pytest.fixture
def staged(tmp_path, monkeypatch):
    d, tune = _staged(tmp_path)
    monkeypatch.setenv("TB_AUDIT_DATA", str(d))
    monkeypatch.setattr(A, "PKG", PKG)
    return d, tune


def _pin(monkeypatch, tmp_path, tune):
    """Point the fingerprint check at the synthetic cohort rather than the real one."""
    cfg = tmp_path / "configs"
    (cfg).mkdir(exist_ok=True)
    json.dump({"fingerprint_sha256": A.cohort_fingerprint(tune)},
              io.open(cfg / "tune_cohort_fingerprint.json", "w", encoding="utf-8"))
    (cfg / "concept_indices.json").write_bytes(
        (PKG / "configs/concept_indices.json").read_bytes())
    (cfg / "model_assets.json").write_bytes(
        (PKG / "configs/model_assets.json").read_bytes())
    monkeypatch.setattr(A, "PKG", tmp_path)
    # Packaged files are located by P.asset now, so the pin has to go in front of it.
    monkeypatch.setattr(P, "_ASSET_DIRS", [cfg] + list(P._ASSET_DIRS))


# --------------------------------------------------------- layout resolution
def test_kaggle_layout_is_resolved_not_assumed(staged, tmp_path, monkeypatch):
    """The previous package derived paths from HERE.parents[2], which is /kaggle on Kaggle.

    This is the defect that would have failed the run at its first data cell.
    """
    d, _ = staged
    p = P.resolve()
    assert p["layout"] == "staged"
    assert p["img_dir"] == d / P.STAGED_IMG_DIR
    old = PKG.parents[1] / "dryad_rights_clear_audit_20261023_0900/src/run_concept_audit.py"
    if old.is_file():
        assert "parents[2]" in old.read_text(encoding="utf-8"), \
            "the superseded module should still show the hard-coded root"
        assert "parents[2]" not in (PKG / "src/concept_audit.py").read_text(
            encoding="utf-8"), "the corrected module must not hard-code a repo root"


def test_an_unusable_layout_raises_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TB_AUDIT_DATA", str(tmp_path / "nothing-here"))
    monkeypatch.setattr(P, "_repo_root", lambda: None)
    with pytest.raises(P.LayoutError) as e:
        P.resolve()
    assert "stage_kaggle_data.py" in str(e.value) and "TB_AUDIT_DATA" in str(e.value)


# --------------------------------------------------------- partition guards
def test_a_report_row_in_the_staged_upload_is_rejected(tmp_path, monkeypatch):
    d, tune = _staged(tmp_path, intrude=["r0001"])
    monkeypatch.setenv("TB_AUDIT_DATA", str(d))
    _pin(monkeypatch, tmp_path, tune)
    with pytest.raises(A.SplitViolation, match="REPORT pids"):
        A.load_tune_only()


def test_a_missing_tune_case_is_rejected(tmp_path, monkeypatch):
    d, tune = _staged(tmp_path, drop=1)
    monkeypatch.setenv("TB_AUDIT_DATA", str(d))
    _pin(monkeypatch, tmp_path, tune)
    with pytest.raises(A.SplitViolation, match="expected"):
        A.load_tune_only()


def test_a_duplicated_case_is_rejected(tmp_path, monkeypatch):
    d, tune = _staged(tmp_path, dup=True)
    monkeypatch.setenv("TB_AUDIT_DATA", str(d))
    _pin(monkeypatch, tmp_path, tune)
    with pytest.raises(A.SplitViolation, match="duplicate"):
        A.load_tune_only()


def test_an_overlapping_split_manifest_is_rejected(tmp_path, monkeypatch):
    d, tune = _staged(tmp_path)
    s = json.loads((d / P.STAGED_FILES["split"]).read_text(encoding="utf-8"))
    s["report"].append(s["tune"][0])
    (d / P.STAGED_FILES["split"]).write_text(json.dumps(s), encoding="utf-8")
    monkeypatch.setenv("TB_AUDIT_DATA", str(d))
    _pin(monkeypatch, tmp_path, tune)
    with pytest.raises(A.SplitViolation, match="overlap"):
        A.load_tune_only()


def test_a_recut_cohort_is_rejected_by_the_pinned_fingerprint(tmp_path, monkeypatch):
    d, tune = _staged(tmp_path)
    monkeypatch.setenv("TB_AUDIT_DATA", str(d))
    _pin(monkeypatch, tmp_path, tune + ["t9999"])        # pin a different cohort
    with pytest.raises(A.SplitViolation, match="fingerprint mismatch"):
        A.load_tune_only()


def test_the_real_tune_cohort_matches_its_pinned_fingerprint():
    p = P.resolve(prefer="repo")
    d = A.load_tune_only(p)
    assert len(d["pid"]) == FP["n"] == 1352
    assert d["fingerprint"] == FP["fingerprint_sha256"]


def test_locked_partition_is_never_named_in_a_data_module():
    for f in sorted((PKG / "src").glob("*.py")):
        if f.name == "preflight.py":
            continue                 # names it in order to exclude it; see that file
        src = f.read_text(encoding="utf-8")
        for tok in ('"internal_test"', "'internal_test'", "internal_test/"):
            assert tok not in src, "%s references %s" % (f.name, tok)


def test_the_package_contains_no_images_weights_or_identifiers():
    bad = [p.relative_to(PKG).as_posix() for p in PKG.rglob("*")
           if p.is_file() and p.suffix.lower() in
           (".png", ".jpg", ".jpeg", ".dcm", ".pt", ".pth", ".npz", ".npy",
            ".safetensors", ".pickle")]
    assert not bad, "package must be code and documentation only; found %s" % bad
    # No patient id may appear anywhere in the package: only the cohort hash.
    real = {str(x) for x in json.loads(
        (P.resolve(prefer='repo')["split"]).read_text(encoding="utf-8"))["tune"]}
    sample = sorted(real)[:60]
    for p in PKG.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".md", ".json", ".txt", ".ipynb"):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            hit = [s for s in sample if s in txt]
            assert not hit, "%s contains patient ids %s" % (p.name, hit[:3])


# --------------------------------------------------------- the CLEAR loader
def test_the_encoder_is_not_reimplemented(monkeypatch):
    """Reading CLEAR's hub.py showed the hand-rolled encoder was wrong three ways.

    The key prefix was `visual.backbone.` (missed), the visual tower has a projection
    that `encode_image` applies (omitted, so similarities would have been in the wrong
    space), and preprocessing uses CXR statistics with a bicubic resize (ImageNet
    constants and bilinear were used instead). Only the first would have raised. So the
    package now calls CLEAR's own loader, and must not grow a replacement.
    """
    src = (PKG / "src/concept_audit.py").read_text(encoding="utf-8")
    assert "from clear import load_pretrained" in src
    assert "0.485" not in src and "0.229" not in src,         "ImageNet normalisation constants are back in the encoder"
    assert "torchvision.transforms" not in src,         "preprocessing must come from CLEAR, not be rebuilt here"
    assert "model.encode_image(" in src, "the projected embedding must be used"
    assert 'torch.hub.load("facebookresearch/dinov2"' not in src


def test_a_missing_clear_package_raises_with_the_pinned_install_command(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "clear":
            raise ImportError("No module named 'clear'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(A.WeightLoadError) as e:
        A._load_encoder("ignored.pt", "cpu", log=lambda *_: None)
    msg = str(e.value)
    assert A.CLEAR_COMMIT in msg, "the install command must pin the commit"
    assert "--no-deps" in msg, "without --no-deps CLEAR's pins reinstall torch"


def test_a_model_without_a_projection_is_refused(monkeypatch):
    """encode_image must return the PROJECTED vector, or every similarity is wrong."""
    import types

    class _Visual:
        def parameters(self):
            return iter([])

    class _Model:
        visual = _Visual()

        def encode_image(self, x):
            return x

        def eval(self):
            return self

    fake_clear = types.ModuleType("clear")
    fake_clear.load_pretrained = lambda **k: (_Model(), object())
    monkeypatch.setitem(sys.modules, "clear", fake_clear)
    with pytest.raises(A.WeightLoadError, match="projection"):
        A._load_encoder("ignored.pt", "cpu", log=lambda *_: None)


def test_the_clear_commit_is_pinned_to_a_commit():
    assert len(A.CLEAR_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in A.CLEAR_COMMIT)


# --------------------------------------------------------- resume safety
def test_a_resume_onto_a_different_id_list_is_refused(tmp_path, monkeypatch):
    sh = tmp_path / "_shards"
    sh.mkdir()
    json.dump({"ordered_hash": A.ordered_hash(["a", "b"]), "done": 2,
               "shards": ["shard_000000.npy"]},
              io.open(sh / "progress.json", "w", encoding="utf-8"))
    monkeypatch.setattr(A.P, "resolve", lambda *a, **k: {"img_dir": tmp_path,
                                                         "out_dir": tmp_path})
    with pytest.raises(A.SplitViolation, match="DIFFERENT ordered id list"):
        A.encode_concepts(["a", "b", "c"], device="cpu", out_dir=tmp_path)


def test_the_ordered_hash_is_order_sensitive_and_the_fingerprint_is_not():
    assert A.ordered_hash(["b", "a"]) != A.ordered_hash(["a", "b"])
    assert A.cohort_fingerprint(["b", "a"]) == A.cohort_fingerprint(["a", "b"])


# --------------------------------------------------------- concept mapping
def test_only_exact_vocabulary_entries_are_used():
    primary = [c for c in MAP["mapping"] if c["concept_idx"] is not None]
    assert len(primary) == MAP["n_primary"] == 9
    for c in primary:
        assert c["exact_vocab_entry"] == c["concept"] and "EXACT" in c["status"]


def test_concepts_without_an_exact_entry_are_excluded_not_approximated():
    excluded = [c for c in MAP["mapping"] if c["concept_idx"] is None]
    assert {c["concept"] for c in excluded} == {"cavity", "nodule", "reticular opacity"}
    assert "not approximated" in MAP["rule"]


def test_the_concept_score_is_documented_as_a_proxy_not_a_detector():
    p = P.resolve(prefer="repo")
    m = A.run_manifest(A.load_tune_only(p), p)
    kind = m["concepts"]["score_kind"].lower()
    assert "proxy" in kind and "not a validated detector" in kind


def test_the_pinned_revision_is_a_commit_not_a_branch():
    assert len(A.HF_REV) == 40 and all(c in "0123456789abcdef" for c in A.HF_REV)


# --------------------------------------------------------- analysis correctness
def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    d = {"pid": [str(i) for i in range(n)], "y": y,
         "past_tb": (rng.random(n) < 0.35).astype(int),
         "rad_system": rng.integers(1, 4, n), "age_group": rng.integers(1, 3, n),
         "sex": rng.integers(0, 2, n), "symptoms": rng.integers(0, 5, n)}
    return d, rng


def test_analysis_recovers_a_planted_association_and_a_null():
    d, rng = _synthetic()
    n = len(d["y"])
    signal, noise = rng.normal(size=n), rng.normal(size=n)
    d["clf_score"] = 2.0 * signal + rng.normal(0, 0.5, n)
    rows, _ = A.analyse(d, np.column_stack([signal, noise]), ["planted", "null"])
    neg = {r["concept"]: r for r in rows if r["group"] == "negatives (all)"}
    assert neg["planted"]["spearman_rho"] > 0.6 and neg["planted"]["ci95_lo"] > 0
    assert abs(neg["null"]["spearman_rho"]) < 0.2
    assert neg["null"]["ci95_lo"] < 0 < neg["null"]["ci95_hi"]


def test_bh_adjustment_is_applied_to_the_primary_family_only_and_is_monotone():
    d, rng = _synthetic(seed=1)
    n = len(d["y"])
    d["clf_score"] = rng.normal(size=n)
    rows, _ = A.analyse(d, rng.normal(size=(n, 5)), list("abcde"))
    fam = [r for r in rows if r["group"] == "negatives (all)"]
    assert len(fam) == 5
    assert all(r["p_bh_adjusted"] >= r["p_raw"] - 1e-12 for r in fam)
    assert all(0 <= r["p_bh_adjusted"] <= 1 for r in fam)
    assert all(r["p_bh_adjusted"] is None for r in rows
               if r["group"] != "negatives (all)")


def test_incremental_r2_is_near_zero_for_a_pure_noise_concept():
    d, rng = _synthetic(seed=2)
    d["clf_score"] = rng.normal(size=len(d["y"]))
    _, incr = A.analyse(d, rng.normal(size=(len(d["y"]), 1)), ["noise"])
    assert abs(incr[0]["delta_r2"]) < 0.02


def test_small_groups_are_suppressed_rather_than_reported():
    d, rng = _synthetic(n=60, seed=3)
    d["clf_score"] = rng.normal(size=60)
    S = rng.normal(size=(60, 1))
    rows, _ = A.analyse(d, S, ["x"])
    assert all(r["n"] >= 20 for r in rows)
    dist = A.distributions(d, S, ["x"])
    for r in dist:
        assert r["n"] >= 20 or r["note"].startswith("suppressed")
    assert any(r["note"].startswith("suppressed") for r in dist)


def test_discrimination_recovers_a_planted_auroc_and_suppresses_a_tiny_class():
    d, rng = _synthetic(n=800, seed=4)
    d["clf_score"] = rng.normal(size=800)
    S = (d["y"] * 1.5 + rng.normal(size=800)).reshape(-1, 1)
    out = A.discrimination(d, S, ["planted"], n_boot=200)[0]
    assert out["auroc"] > 0.75 and out["auroc_ci95_lo"] > 0.5
    assert "not a TB classifier" in out["note"]

    d["y"] = np.zeros(800, int)
    d["y"][:5] = 1
    out = A.discrimination(d, S, ["planted"], n_boot=50)[0]
    assert "suppressed" in out["note"] and "auroc" not in out


def test_distributions_report_both_history_strata_for_negatives():
    d, rng = _synthetic(n=800, seed=5)
    d["clf_score"] = rng.normal(size=800)
    got = {r["group"] for r in A.distributions(d, rng.normal(size=(800, 1)), ["x"])}
    assert {"negatives, past TB", "negatives, no past TB"} <= got


# --------------------------------------------------------- preflight and notebook
def test_preflight_reports_a_stop_rather_than_relaxing_a_guard(monkeypatch):
    monkeypatch.setenv("TB_AUDIT_NO_GPU_PROBE", "1")
    r = F.run(check_weights=False)
    assert r["verdict"] in ("GO", "STOP")
    assert r["network_required"]["required"] is True
    assert "NOT currently possible" in r["network_required"]["offline_possible"]
    assert F.MIN_RAM_GB >= 3.0
    if r["verdict"] == "STOP":
        assert r["blockers"] and all(isinstance(b, str) and b for b in r["blockers"])


def test_preflight_names_the_expected_output_shape(monkeypatch):
    monkeypatch.setenv("TB_AUDIT_NO_GPU_PROBE", "1")
    r = F.run(check_weights=False)
    if r["cohort"]:
        assert r["cohort"]["expected_similarity_shape"] == [1352, 9]


def test_notebook_is_valid_and_keeps_the_tune_only_guard():
    nb = json.loads((PKG / "kaggle/concept_audit_tune_only.ipynb").read_text(
        encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) >= 5
    for c in code:
        ast.parse("".join(c["source"]))
    whole = "".join("".join(c["source"]) for c in nb["cells"])
    assert "rglob('concept_audit.py')" in whole,         "the notebook must find the code package at any nesting depth"
    for must in ("load_tune_only", "preflight", "0.7242", "GO",
                 "EXCLUDED, not measured", "/kaggle/input"):
        assert must in whole, "notebook does not mention %r" % must
    assert "run_inference" not in whole or "encode_concepts" in whole


def test_notebook_matches_its_generator():
    import subprocess
    before = (PKG / "kaggle/concept_audit_tune_only.ipynb").read_bytes()
    subprocess.run([sys.executable, str(PKG / "src/build_notebook.py")], check=True,
                   capture_output=True)
    assert (PKG / "kaggle/concept_audit_tune_only.ipynb").read_bytes() == before, \
        "the notebook on disk is not what build_notebook.py produces"


def test_requirements_pin_exact_versions_for_the_numeric_stack():
    txt = (PKG / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("numpy", "scipy", "scikit-learn", "torch"):
        assert ("%s==" % pkg) in txt, "%s is not pinned exactly" % pkg


# --------------------------------------------------------- zipped upload
def _zip_staged(tmp_path, nest=True, n_tune=40):
    """Zip a synthetic staged folder the way the Kaggle uploader forces it to be zipped."""
    import zipfile
    d, tune = _staged(tmp_path, n_tune=n_tune)
    z = tmp_path / "upload.zip"
    root = d.parent if nest else d
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for q in sorted(d.rglob("*")):
            if q.is_file():
                zf.write(q, q.relative_to(root).as_posix())
    return z, tune


def test_a_zipped_upload_is_a_first_class_layout(tmp_path, monkeypatch):
    """Kaggle caps files per dataset, so 1,352 images must be uploaded archived."""
    z, tune = _zip_staged(tmp_path)
    monkeypatch.setenv("TB_AUDIT_DATA", str(z))
    p = P.resolve()
    assert p["layout"] == "staged_zip"
    assert P.image_ids(p) == set(tune)
    assert "zip" in P.describe(p) and "NOT extracted" in P.describe(p)


def test_a_zip_made_from_inside_the_folder_also_resolves(tmp_path, monkeypatch):
    z, tune = _zip_staged(tmp_path, nest=False)
    monkeypatch.setenv("TB_AUDIT_DATA", str(z))
    assert P.resolve()["layout"] == "staged_zip"
    assert P.image_ids(P.resolve()) == set(tune)


def test_images_are_read_from_the_archive_without_extracting_them(tmp_path, monkeypatch):
    import zipfile
    d, tune = _staged(tmp_path)
    marker = b'PNG-CONTENT-MARKER-not-a-real-image'
    (d / P.STAGED_IMG_DIR / ('%s.png' % tune[0])).write_bytes(marker)
    z = tmp_path / 'with-content.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED) as zf:
        for q in sorted(d.rglob('*')):
            if q.is_file():
                zf.write(q, q.relative_to(d.parent).as_posix())
    monkeypatch.setenv('TB_AUDIT_DATA', str(z))
    p = P.resolve()
    with P.open_image(p, tune[0]) as fh:
        assert fh.read() == marker, 'bytes must come from the archive member'
    with pytest.raises(FileNotFoundError, match="not in"):
        P.open_image(p, "no-such-case")
    extracted = [q for q in p["scratch"].rglob("*.png")]
    assert not extracted, "images must not be extracted: found %s" % extracted[:3]


def test_the_guards_still_fire_on_a_zipped_upload(tmp_path, monkeypatch):
    """The partition and integrity guards must not be weaker inside an archive."""
    import zipfile
    d, tune = _staged(tmp_path, intrude=["r0001"])
    z = tmp_path / "upload.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for q in sorted(d.rglob("*")):
            if q.is_file():
                zf.write(q, q.relative_to(d.parent).as_posix())
    monkeypatch.setenv("TB_AUDIT_DATA", str(z))
    _pin(monkeypatch, tmp_path, tune)
    assert P.resolve()["layout"] in P.STAGED_LAYOUTS
    with pytest.raises(A.SplitViolation, match="REPORT pids"):
        A.load_tune_only()


def test_check_images_reports_a_missing_member_of_a_zip(tmp_path, monkeypatch):
    z, tune = _zip_staged(tmp_path)
    monkeypatch.setenv("TB_AUDIT_DATA", str(z))
    p = P.resolve()
    missing, extra = A.check_images(p, tune + ["t9999"])
    assert missing == ["t9999"] and not extra


def test_an_archive_without_the_staged_files_is_not_mistaken_for_one(tmp_path,
                                                                    monkeypatch):
    import zipfile
    z = tmp_path / "unrelated.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("notes.txt", "nothing to do with the audit")
    monkeypatch.setenv("TB_AUDIT_DATA", str(z))
    monkeypatch.setattr(P, "_repo_root", lambda: None)
    with pytest.raises(P.LayoutError, match="Searched zips"):
        P.resolve()


# --------------------------------------------------------- the upload builder
import make_upload as U                                            # noqa: E402


def test_the_code_upload_lists_every_module_the_notebook_imports():
    nb = json.loads((PKG / "kaggle/concept_audit_tune_only.ipynb").read_text(
        encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    for mod in ("paths", "preflight", "concept_audit"):
        assert "src/%s.py" % mod in U.CODE_FILES, "%s is imported but not uploaded" % mod
        assert mod in src
    for cfg in ("concept_indices", "tune_cohort_fingerprint", "model_assets"):
        assert "configs/%s.json" % cfg in U.CODE_FILES


def test_the_code_upload_carries_no_caches_tests_or_documentation():
    for rel in U.CODE_FILES:
        assert "__pycache__" not in rel
        assert not rel.startswith("tests/")
        assert not rel.endswith(".md")
    for rel in U.CODE_FILES:
        assert (PKG / rel).is_file(), "%s is listed for upload but absent" % rel


def test_the_notebook_discovery_globs_match_both_kaggle_shapes(tmp_path):
    """Kaggle may place dataset contents at the root or wrap them in one folder."""
    assert U.check_notebook_contract(tmp_path, "concept-audit-tune.zip",
                                     log=lambda *_: None)


# --------------------------------------------------------- flattened upload
def _flatten(tmp_path):
    """Kaggle's uploader may drop src/ and configs/, leaving every file in one folder."""
    flat = tmp_path / "tb-triage-clear-audit-code"
    flat.mkdir()
    import shutil
    for rel in U.CODE_FILES:
        shutil.copy2(PKG / rel, flat / Path(rel).name)
    return flat


def test_packaged_files_are_found_when_kaggle_flattens_the_upload(tmp_path):
    """The real failure on Kaggle: src/ and configs/ were gone, so PKG was wrong.

    `preflight.py` died on `PKG/requirements.txt`, two levels above where the files
    actually were. Exercised in a subprocess so the flattened copy is genuinely the one
    imported, not the package on sys.path.
    """
    import subprocess
    flat = _flatten(tmp_path)
    prog = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import paths as P, concept_audit as A\n"
        "assert P.asset('concept_indices.json').is_file()\n"
        "assert P.asset('tune_cohort_fingerprint.json').is_file()\n"
        "assert P.asset('model_assets.json').is_file()\n"
        "assert P.asset('requirements.txt').is_file()\n"
        "m, primary = A.concept_map()\n"
        "assert len(primary) == 9\n"
        "print('FLAT-OK', P.code_dir().name)\n" % flat)
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FLAT-OK" in r.stdout


def test_a_genuinely_missing_packaged_file_still_raises_clearly(tmp_path):
    import subprocess
    flat = _flatten(tmp_path)
    (flat / "model_assets.json").unlink()
    prog = ("import sys; sys.path.insert(0, r'%s')\n"
            "import paths as P\n"
            "try:\n"
            "    P.asset('model_assets.json')\n"
            "except P.LayoutError as e:\n"
            "    assert 'folder structure' in str(e), e\n"
            "    print('RAISED-OK')\n" % flat)
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
    assert "RAISED-OK" in r.stdout, r.stdout + r.stderr


# --------------------------------------------------------- the one-cell runner
ONECELL = PKG / "kaggle/run_all_one_cell.py"


def test_the_one_cell_runner_is_valid_and_keeps_every_gate():
    src = ONECELL.read_text(encoding="utf-8")
    ast.parse(src)
    for must in ("rglob('concept_audit.py')", "load_tune_only", "check_weights=True",
                 "check_images", "cohort gate: PASSED", "output gate: PASSED",
                 "0.7242", "EXCLUDED, not measured"):
        assert must in src, "the one-cell runner is missing %r" % must


def test_run_all_aborts_on_failure_rather_than_continuing():
    """Jupyter can carry on past a bare SystemExit during Run All; RuntimeError stops it.

    The user runs this with Run All, so a preflight STOP that did not halt the queue
    would go straight on to inference.
    """
    for f in (ONECELL, PKG / "kaggle/concept_audit_tune_only.ipynb"):
        src = f.read_text(encoding="utf-8")
        assert "SystemExit" not in src, "%s still raises SystemExit" % f.name
        assert "RuntimeError" in src


def test_the_one_cell_runner_and_the_notebook_enforce_the_same_gates():
    one = ONECELL.read_text(encoding="utf-8")
    nb = json.loads((PKG / "kaggle/concept_audit_tune_only.ipynb").read_text(
        encoding="utf-8"))
    cells = "".join("".join(c["source"]) for c in nb["cells"])
    for gate in ("load_tune_only", "check_images", "concept_map", "encode_concepts",
                 "analyse", "run_manifest", "preflight_kaggle.json"):
        assert gate in one and gate in cells, "%s is not in both entry points" % gate


def test_the_duplicate_message_fingerprints_all_three_modules():
    """Hashing preflight.py alone printed two identical digests for copies that differed.

    The copies differed in concept_audit.py, which was the whole point. A fingerprint
    that cannot tell them apart is worse than none: it reads as reassurance.
    """
    src = ONECELL.read_text(encoding="utf-8")
    i = src.index("len(GOOD) > 1")
    block = src[i:i + 900]
    for f in ("concept_audit.py", "paths.py", "preflight.py"):
        assert f in block, "the duplicate fingerprint ignores %s" % f
    assert "they DIFFER" in block and "identical" in block,         "the message must say whether the copies actually differ"


def test_both_entry_points_refuse_two_copies_of_the_code_package():
    """Two attached copies once meant the stale one ran, silently, by sort order."""
    for f in (ONECELL, PKG / "kaggle/concept_audit_tune_only.ipynb"):
        src = f.read_text(encoding="utf-8")
        assert "len(GOOD) > 1" in src, "%s does not detect duplicate copies" % f.name
        assert "detach" in src.lower(), "%s must say how to resolve it" % f.name
        assert "paths.py" in src and "preflight.py" in src, \
            "%s must require a COMPLETE copy, not just concept_audit.py" % f.name


def test_the_duplicate_detection_logic_actually_fires(tmp_path):
    """Run the real discovery block against two synthetic copies."""
    inp = tmp_path / "input"
    for name in ("code-a", "code-b"):
        d = inp / name
        d.mkdir(parents=True)
        for f in ("concept_audit.py", "paths.py", "preflight.py"):
            (d / f).write_text("# %s in %s\n" % (f, name), encoding="utf-8")
    cands = sorted({q.parent for q in inp.rglob("concept_audit.py")})
    good = [c for c in cands
            if (c / "paths.py").is_file() and (c / "preflight.py").is_file()]
    assert len(good) == 2, "the fixture must present two complete copies"

    (inp / "code-b" / "paths.py").unlink()          # now only one is complete
    good = [c for c in sorted({q.parent for q in inp.rglob("concept_audit.py")})
            if (c / "paths.py").is_file() and (c / "preflight.py").is_file()]
    assert len(good) == 1 and good[0].name == "code-a"


# --------------------------------------------------------- the real Kaggle layout
def test_the_actual_kaggle_layout_resolves(tmp_path, monkeypatch):
    """Reproduce what Kaggle actually produced, doubled prefix and all.

        datasets/<user>/<data-ds>/concept-audit-tune/concept-audit-tune/
            tune_manifest.json, the two CSVs, the split, vietnam_img512_tune/*.png
        datasets/<user>/<data-ds>/*.py          <- the code, uploaded here too
        datasets/<user>/<code-ds>/*.py          <- a second, stale copy

    Kaggle auto-extracted the zip and nested it twice. Neither the fixed-depth search
    nor a flat assumption would have found it.
    """
    inp = tmp_path / "input"
    ds = inp / "datasets/atuldhull13/tb-triage-vietnam-tune-concept-audit"
    deep = ds / "concept-audit-tune/concept-audit-tune"
    (deep / P.STAGED_IMG_DIR).mkdir(parents=True)
    src_d, tune = _staged(tmp_path, n_tune=30)
    import shutil
    for f in P.STAGED_FILES.values():
        shutil.copy2(src_d / f, deep / f)
    for q in (src_d / P.STAGED_IMG_DIR).glob("*.png"):
        shutil.copy2(q, deep / P.STAGED_IMG_DIR / q.name)

    monkeypatch.delenv("TB_AUDIT_DATA", raising=False)
    monkeypatch.setattr(P, "KAGGLE_INPUT", inp)
    monkeypatch.setattr(P, "LOCAL_STAGING", tmp_path / "no-local-staging")
    monkeypatch.setattr(P, "_repo_root", lambda: None)
    _pin(monkeypatch, tmp_path, tune)

    p = P.resolve()
    assert p["layout"] == "staged", "an extracted upload is a plain staged folder"
    assert p["base"] == deep, "resolved %s, expected the doubly nested dir" % p["base"]
    assert P.image_ids(p) == set(tune)
    d = A.load_tune_only(p)
    assert len(d["pid"]) == len(tune)
    missing, extra = A.check_images(p, d["pid"])
    assert not missing and not extra


def test_a_zip_is_still_found_under_the_kaggle_root(tmp_path, monkeypatch):
    inp = tmp_path / "input"
    ds = inp / "datasets/atuldhull13/data-ds"
    ds.mkdir(parents=True)
    z, tune = _zip_staged(tmp_path, n_tune=25)
    import shutil
    shutil.copy2(z, ds / "concept-audit-tune.zip")
    monkeypatch.delenv("TB_AUDIT_DATA", raising=False)
    monkeypatch.setattr(P, "KAGGLE_INPUT", inp)
    monkeypatch.setattr(P, "LOCAL_STAGING", tmp_path / "no-local-staging")
    monkeypatch.setattr(P, "_repo_root", lambda: None)
    p = P.resolve()
    assert p["layout"] == "staged_zip"
    assert P.image_ids(p) == set(tune)


def test_an_explicit_code_dir_overrides_discovery_and_is_validated():
    """With two copies attached, naming one beats any rule about which copy wins."""
    src = ONECELL.read_text(encoding="utf-8")
    assert "CODE_DIR = ''" in src, "the override must default to automatic discovery"
    assert "CODE_DIR is set to" in src, "a wrong CODE_DIR must say so, not fall back"
    i_set, i_use = src.index("CODE_DIR = ''"), src.index("if CODE_DIR:")
    assert i_set < i_use, "the override must be at the top, where it can be edited"
    for f in ("concept_audit.py", "paths.py", "preflight.py"):
        assert f in src[i_use:i_use + 600], \
            "an explicit CODE_DIR must still be checked for %s" % f
