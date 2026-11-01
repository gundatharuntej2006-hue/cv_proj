# -*- coding: utf-8 -*-
"""Focused tests for the experimental code written in this run.

These stay in the run directory. They do not touch production code or the production
test suite.

    py -3 -m pytest artifacts/overnight_scanner_run_20261021_0208/tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(RUN / "scripts"))

import common as C                                                   # noqa: E402

torch = pytest.importorskip("torch")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _free_vram():
    """Loader tests leave tensors on the device; release them so later tests can allocate."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ loader
def test_npy_header_matches_numpy():
    import zipfile
    p = C.REG / "regional_nlm_lungs_g12_v1.npz"
    with zipfile.ZipFile(p) as zf:
        shape, dtype, fortran = C.npy_header(zf, "reg.npy")
    assert shape == (800, 144, 768)
    assert np.dtype(dtype) == np.float16
    assert fortran is False


def test_stream_rejects_a_grid_that_is_not_12x12():
    """A silent grid mismatch already cost this project once; it must raise now."""
    p = C.REG / "regional_tbx11k_lungs_g18_v1.npz"
    if not p.exists():
        pytest.skip("18-grid cache not present")
    with pytest.raises(ValueError, match="regions"):
        C.stream_regions(p, keep_idx=[0], device=DEV, verbose=False)


def test_streamed_rows_are_l2_normalised():
    x, _ = C.stream_regions(C.REG / "regional_nlm_lungs_g12_v1.npz",
                            keep_idx=list(range(16)), device=DEV, verbose=False)
    n = x.float().norm(dim=2)
    assert float(n.min()) > 0.99 and float(n.max()) < 1.01


def test_keep_idx_selects_the_rows_it_says_it_does():
    """Selecting a scattered subset must return exactly those rows, in sorted order."""
    want = [3, 1, 700, 55]
    x, kept = C.stream_regions(C.REG / "regional_nlm_lungs_g12_v1.npz",
                               keep_idx=want, device=DEV, verbose=False)
    assert list(kept) == sorted(want)
    full, _ = C.stream_regions(C.REG / "regional_nlm_lungs_g12_v1.npz",
                               keep_idx=sorted(want), block=1, device=DEV, verbose=False)
    assert torch.equal(x, full), "block size must not change which rows are returned"


# ------------------------------------------------------------------ aggregation
def test_agg_matches_the_committed_implementation():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import p2_32_mil_sweep as S
    rng = np.random.default_rng(0)
    s = rng.standard_normal((32, 144)).astype(np.float32)
    for how in ("mean", "max", "top3", "top5"):
        mine = C.agg(torch.from_numpy(s), how).numpy()
        theirs = S.agg_np(s, how)
        assert np.allclose(mine, theirs, atol=1e-5), how


# ------------------------------------------------------------------ projection
def test_project_out_removes_exactly_the_named_subspace():
    rng = np.random.default_rng(0)
    B = np.linalg.qr(rng.standard_normal((768, 3)))[0].T.astype(np.float32)
    Bt = torch.from_numpy(B)
    x = torch.from_numpy(rng.standard_normal((8, 144, 768)).astype(np.float16))
    sys.path.insert(0, str(RUN / "scripts"))
    import task3_subspace_tune as T3
    T3.project_out(x, Bt)
    resid = (x.float().reshape(-1, 768) @ Bt.T).abs().max()
    assert float(resid) < 2e-2, "projected features retain a component in the subspace"


def test_projection_is_idempotent():
    rng = np.random.default_rng(1)
    B = np.linalg.qr(rng.standard_normal((768, 3)))[0].T.astype(np.float32)
    Bt = torch.from_numpy(B)
    x = torch.from_numpy(rng.standard_normal((4, 144, 768)).astype(np.float16))
    import task3_subspace_tune as T3
    T3.project_out(x, Bt)
    once = x.clone()
    T3.project_out(x, Bt)
    assert torch.allclose(once.float(), x.float(), atol=1e-2)


# ------------------------------------------------------------------ split hygiene
def test_tune_and_report_do_not_overlap():
    _, tune, report = C.vietnam_meta()
    assert len(tune) == 1352 and len(report) == 1356
    assert not (tune & report)


def test_every_tune_only_artifact_declares_report_untouched():
    for name in ("tables/thresholds_tune.json", "tables/subspace_tune.json",
                 "tables/representation_diagnostics.json", "tables/inlp_diagnostics.json",
                 "tables/offset_vs_ranking.json"):
        p = RUN / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        flag = d.get("report_touched", d.get("report_metrics_computed"))
        assert flag is False, "%s does not declare REPORT untouched" % name


def test_frozen_parameters_predate_the_sealed_report_pass():
    man = RUN / "FROZEN_MANIFEST.md"
    par = RUN / "tables/frozen_parameters.json"
    sealed = RUN / "tables/acceptance_verdict.json"
    assert man.exists() and par.exists()
    if sealed.exists():
        assert par.stat().st_mtime <= sealed.stat().st_mtime
        assert man.stat().st_mtime <= sealed.stat().st_mtime


def test_offset_correction_cannot_change_within_scanner_ranking():
    """A per-scanner constant shift is rank-preserving inside each scanner, by construction.

    If a future edit broke this, per-scanner AUROC would move and the offset arm's whole
    interpretation would change, so it is pinned here.
    """
    import csv
    import io
    from sklearn.metrics import roc_auc_score
    f = RUN / "tables/vietnam_case_scores.csv"
    if not f.exists():
        pytest.skip("scores not produced yet")
    rows = list(csv.DictReader(io.open(f, encoding="utf-8-sig")))
    y = np.array([int(r["xpert_result"]) for r in rows])
    s = np.array([float(r["score_baseline"]) for r in rows])
    g = np.array([int(r["rad_system"]) for r in rows])
    P = json.loads((RUN / "tables/frozen_parameters.json").read_text(encoding="utf-8"))
    sc = s.copy()
    for k in (1, 2, 3):
        sc[g == k] -= P["scanner_offsets_tune_non_tb_median"][str(k)]
    for k in (1, 2, 3):
        m = g == k
        assert abs(roc_auc_score(y[m], s[m]) - roc_auc_score(y[m], sc[m])) < 1e-12
