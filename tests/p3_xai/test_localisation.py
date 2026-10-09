"""P3-10: localisation metrics (synthetic) and the development C11 export (stand-in classifiers, fake P1 data)."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "tests" / "p3_mock"))

from p3_xai.metrics import box_mask, energy_in_box, localisation_row, peak, pointing_hit  # noqa: E402

warnings.filterwarnings("ignore", message="D21 not locked")
warnings.filterwarnings("ignore", message="no coco_")
warnings.filterwarnings("ignore", message="D14 model registry")


def hm_with_peak(x, y):
    h = np.zeros((512, 512), np.float32)
    h[y, x] = 1.0
    return h


# --- pointing --------------------------------------------------------------------------------------

def test_pointing_peak_inside_box():
    assert pointing_hit(hm_with_peak(150, 120), [[100, 100, 200, 150]]) is True


def test_pointing_peak_outside_box():
    assert pointing_hit(hm_with_peak(250, 120), [[100, 100, 200, 150]]) is False


def test_pointing_any_box_counts():
    assert pointing_hit(hm_with_peak(320, 330), [[100, 100, 200, 150], [300, 300, 350, 360]]) is True


def test_pointing_pixel_centre_convention_at_box_edges():
    box = [[100, 100, 200, 150]]              # pixels 100..199 x 100..149 have centres inside
    assert pointing_hit(hm_with_peak(100, 100), box) and pointing_hit(hm_with_peak(199, 149), box)
    assert not pointing_hit(hm_with_peak(200, 120), box) and not pointing_hit(hm_with_peak(99, 120), box)


def test_pointing_first_maximum_and_empty_heatmap():
    h = np.zeros((512, 512), np.float32)
    h[10, 300] = h[400, 20] = 0.9               # tie: first in row-major order (y=10)
    assert peak(h) == (300, 10)
    assert pointing_hit(np.zeros((512, 512)), [[0, 0, 512, 512]]) is False   # no heat, no peak


# --- energy in box -----------------------------------------------------------------------------------

def test_energy_uniform_heatmap_equals_box_area_share():
    h = np.ones((512, 512), np.float16)
    assert energy_in_box(h, [[100, 100, 200, 150]]) == pytest.approx(100 * 50 / 512 ** 2, abs=1e-7)
    # fractional box: within one pixel row/column of the exact area
    e = float(energy_in_box(h, [[100.3, 100.6, 220.8, 170.2]]))
    exact = (220.8 - 100.3) * (170.2 - 100.6) / 512 ** 2
    assert abs(e - exact) < (120.5 + 69.6) / 512 ** 2


def test_energy_all_mass_inside_box_is_one():
    h = np.zeros((512, 512), np.float32)
    h[110:140, 120:180] = np.random.default_rng(0).random((30, 60))
    e = energy_in_box(h, [[100, 100, 200, 150]])
    assert e == pytest.approx(1.0) and e.dtype == np.float32


def test_energy_union_counts_overlap_once_and_zero_mass():
    h = np.ones((512, 512), np.float32)
    one = energy_in_box(h, [[100, 100, 200, 200]])
    assert energy_in_box(h, [[100, 100, 200, 200], [100, 100, 200, 200]]) == one
    assert energy_in_box(np.zeros((512, 512)), [[100, 100, 200, 200]]) == 0.0
    assert box_mask([[0, 0, 512, 512]]).all()


def test_localisation_row_only_for_images_with_boxes():
    with pytest.raises(ValueError, match="only on TB images"):
        localisation_row("h0001", "m", np.ones((512, 512)), [], None)
    r = localisation_row("tb0001", "m", hm_with_peak(150, 120), [[100, 100, 200, 150]], "active")
    assert (r.peak_x, r.peak_y, r.pointing_hit, r.n_gt_boxes) == (150, 120, True, 1)


# --- development export: two models -------------------------------------------------------------------

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pytorch_grad_cam")
pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from standin import StandInRadDino

    from p3_det.data import DataPaths
    from p3_det.data.fake_p1 import make_fake_p1_tree
    from p3_xai import export_c11 as ex
    tmp = tmp_path_factory.mktemp("p3_10")
    root = make_fake_p1_tree(tmp / "p1", n_per_split=20)
    reg = tmp / "p2_model_registry.json"
    reg.write_text(json.dumps({"value": {"models": [{"model_id": "radino_crop_m0.10_clahe0_bin"},
                                                    {"model_id": "radino_full_clahe0_bin"}]}}))
    runs = [ex.ModelRun("radino_crop_m0.10_clahe0_bin", "crop", lambda: StandInRadDino(seed=1)),
            ex.ModelRun("radino_full_clahe0_bin", "full", lambda: StandInRadDino(seed=2))]
    s = ex.export(runs, paths=DataPaths(root), out_root=tmp / "xai", decisions_dir=tmp / "nodec", registry=reg)
    return {"s": s, "tmp": tmp, "root": root, "ex": ex, "out": tmp / "xai"}


def test_c11_development_files_validate(exported):
    import standin_contracts as sc

    from p3_xai import c11
    out = exported["out"]
    for mid in ("radino_crop_m0.10_clahe0_bin", "radino_full_clahe0_bin"):
        p = out / f"loc_{mid}_development_v1.parquet"
        d = c11.validate_table(p).to_pydict()
        sc.validate_c11_table(p)                            # P3-03 stand-in validator too
        assert len(d["image_id"]) > 0 and set(d["model_id"]) == {mid}
        assert set(d["tb_subtype"]) <= {"active", "latent", "active+latent"} and min(d["n_gt_boxes"]) > 0
        for iid in d["image_id"]:
            assert (out / mid / "development" / f"{iid}.npy").exists()


def test_rows_are_development_tb_images_with_boxes(exported):
    import csv

    import pyarrow.parquet as pq
    with open(exported["root"] / "data/splits/tbx11k_splits_v1.csv", newline="") as f:
        c01 = {r["image_id"]: r for r in csv.DictReader(f)}
    d = pq.read_table(exported["out"] / "loc_radino_crop_m0.10_clahe0_bin_development_v1.parquet").to_pydict()
    expected = {i for i, r in c01.items() if r["split"] == "development" and r["tb_label"] == "1" and r["has_boxes"] == "True"}
    assert set(d["image_id"]) == expected
    assert set(d["tb_subtype"]) == {"active"}                 # fake C01 category active_tb -> "active"


def test_crop_model_zero_outside_crop_full_model_uses_whole_image(exported):
    from p3_xai import c11
    out = exported["out"]
    items = {i["image_id"]: i for i in exported["ex"].development_items(
        __import__("p3_det.data", fromlist=["DataPaths"]).DataPaths(exported["root"]), exported["tmp"] / "nodec")}
    iid = next(iter(items))
    c11.validate_heatmap(out / "radino_crop_m0.10_clahe0_bin" / "development" / f"{iid}.npy", items[iid]["crop_box"])
    full = np.load(out / "radino_full_clahe0_bin" / "development" / f"{iid}.npy")
    x0, y0, x1, y1 = items[iid]["crop_box"]
    outside = np.ones((512, 512), bool)
    outside[y0:y1, x0:x1] = False
    assert full[outside].any()                                # full-image model sees outside the lung crop


def test_summary_has_n_and_cis_for_both_models_and_paired_difference(exported, capsys):
    s = exported["s"]
    assert set(s["models"]) == {"radino_crop_m0.10_clahe0_bin", "radino_full_clahe0_bin"}
    for r in s["models"].values():
        for k in ("pointing_accuracy", "mean_energy_in_box"):
            m = r[k]
            assert m["n_images"] > 0 and m["n_boxes"] >= m["n_images"] and m["ci_method"] == "bootstrap_percentile"
            assert m["ci_low"] <= m["value"] <= m["ci_high"]
    assert s["pointing_accuracy_crop_minus_full"]["n_images"] == s["models"]["radino_full_clahe0_bin"]["pointing_accuracy"]["n_images"]
    assert s["registry_check"] == "D14"
    exported["ex"].print_summary(s)
    out = capsys.readouterr().out
    assert "n img" in out and "[" in out and "not a finding" in out
    assert (exported["out"] / "loc_summary_development.json").exists()


def test_registry_refuses_unknown_model(exported, tmp_path):
    from p3_det import decisions as dec
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"value": {"models": [{"model_id": "radino_crop_m0.10_clahe0_bin"}]}}))
    with pytest.raises(dec.Refused, match="radino_full"):
        exported["ex"].check_registry(["radino_crop_m0.10_clahe0_bin", "radino_full_clahe0_bin"], reg)
    assert exported["ex"].check_registry(["x"], tmp_path / "missing.json") == "unverified (no D14)"


def test_p2_provider_reports_missing_r2(exported):
    if __import__("importlib").util.find_spec("p2_cls") is not None:
        pytest.skip("p2_cls present")
    with pytest.raises(RuntimeError, match="classifier_module"):
        exported["ex"].p2_provider("radino_full_clahe0_bin", "full")()
