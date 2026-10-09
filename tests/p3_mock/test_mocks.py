"""P3-03: every P3 mock (C10, C11, C15) validates against the (stand-in) contract validators."""
from __future__ import annotations

import json
import re
import warnings

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("jsonschema")
pytest.importorskip("pyarrow")

import make_c10  # noqa: E402
import make_c11  # noqa: E402
import make_c15  # noqa: E402
import standin_contracts as sc  # noqa: E402
from _p3_common import fake_crop_box, mock_image_ids  # noqa: E402


@pytest.fixture(scope="module")
def mock_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mock")
    return {
        "root": root,
        "c10": make_c10.generate("tbx11k", "development", n=30, seed=0, root=root),
        "c11": make_c11.generate("development", n=12, seed=0, root=root),
        "c15": make_c15.generate(n=40, seed=0, root=root),
    }


def test_tbcore_switch_reminder():
    try:
        import tbcore.validators  # noqa: F401
    except ImportError:
        pytest.skip("tbcore not available yet; using stand-in validators")
    warnings.warn("tbcore.validators exists now: switch tests/p3_mock to it (P3-03 follow-up)")


# --- C10 ---------------------------------------------------------------------------------------

def test_c10_validates_and_sits_at_contract_path(mock_root):
    path = mock_root["c10"]
    assert path == mock_root["root"] / "artifacts/p3/det/detections_tbx11k_development_v1.json"
    doc = sc.validate_c10(path)
    assert doc["coords"] == "full512" and doc["schema_version"] == "1.0"
    assert len(doc["images"]) == 30


def test_c10_boxes_inside_crop_and_low_scores_kept(mock_root):
    doc = json.loads(mock_root["c10"].read_text())
    all_scores = []
    for im in doc["images"]:
        x0, y0, x1, y1 = im["crop_box"]
        for bx0, by0, bx1, by1, s in im["boxes"]:
            assert x0 <= bx0 and y0 <= by0 and bx1 <= x1 and by1 <= y1
            all_scores.append(s)
        assert [b[4] for b in im["boxes"]] == sorted((b[4] for b in im["boxes"]), reverse=True)
    assert min(all_scores) < doc["conf_threshold"] <= max(all_scores)   # sub-threshold boxes kept
    assert any(im["boxes"] == [] for im in doc["images"])              # empty images represented
    assert any(im["n_lesions_above_thr"] > 0 for im in doc["images"])


def test_c10_deterministic(tmp_path):
    a = make_c10.generate(n=5, seed=3, root=tmp_path / "a").read_text()
    b = make_c10.generate(n=5, seed=3, root=tmp_path / "b").read_text()
    assert a == b


def test_c10_validator_catches_errors(mock_root, tmp_path):
    doc = json.loads(mock_root["c10"].read_text())
    bad = tmp_path / "detections_tbx11k_development_v1.json"
    for mutate in (lambda d: d.update(coords="crop"),
                   lambda d: d["images"][0].update(n_lesions_above_thr=99),
                   lambda d: d["images"][0].update(boxes=[[0, 0, 10, 10, 0.001]], max_score=0.001,
                                                   n_lesions_above_thr=0)):
        d = json.loads(json.dumps(doc))
        mutate(d)
        bad.write_text(json.dumps(d))
        with pytest.raises((AssertionError, Exception)):
            sc.validate_c10(bad)


# --- C11 ---------------------------------------------------------------------------------------

def test_c11_heatmaps_validate(mock_root):
    xai = mock_root["root"] / "artifacts/p3/xai" / make_c11.c.CAM_MODEL_ID / "development"
    ids = mock_image_ids(12, 0)
    assert sorted(p.stem for p in xai.glob("*.npy")) == sorted(ids)
    for image_id in ids:
        h = sc.validate_c11_heatmap(xai / f"{image_id}.npy", crop_box=fake_crop_box(image_id, 0))
        assert h.max() == 1.0


def test_c11_table_validates_and_matches_heatmaps(mock_root):
    path = mock_root["c11"]
    assert path.name == f"loc_{make_c11.c.CAM_MODEL_ID}_development_v1.parquet"
    d = sc.validate_c11_table(path).to_pydict()
    xai = path.parent / make_c11.c.CAM_MODEL_ID / "development"
    for i, image_id in enumerate(d["image_id"]):
        h = np.load(xai / f"{image_id}.npy").astype(np.float32)
        assert h[d["peak_y"][i], d["peak_x"][i]] == h.max()
        assert d["n_gt_boxes"][i] > 0 and d["tb_subtype"][i] in ("active", "latent", "active+latent")
    assert any(d["pointing_hit"])                      # mock heat is centred on the first lesion


def test_c10_and_c11_mocks_agree_on_image_ids(tmp_path):
    c10 = json.loads(make_c10.generate(n=9, root=tmp_path).read_text())
    c11 = sc.validate_c11_table(make_c11.generate(n=9, root=tmp_path)).to_pydict()
    tb_ids = [im["image_id"] for im in c10["images"] if im["image_id"].startswith("tb")]
    assert tb_ids == c11["image_id"]                       # C11 rows: TB images with boxes only


# --- C15 ---------------------------------------------------------------------------------------

def test_c15_validates(mock_root):
    lines = sc.validate_c15_file(mock_root["c15"])
    assert mock_root["c15"] == mock_root["root"] / "logs/predictions.jsonl"
    statuses = {l["status"] for l in lines}
    assert statuses == {"ANALYSED", "REJECTED_GATE", "UNREADABLE"}
    assert {l["operating_point"] for l in lines} == {"locked_default", "user_override"}
    assert any(l["link_fired"] for l in lines)
    # link rule only ever moves NOT_TB -> REFER
    assert all(l["category_final"] == "REFER" for l in lines if l["link_fired"])
    ts = [l["ts"] for l in lines]
    assert ts == sorted(ts)


def test_c15_has_no_identifier_like_content(mock_root):
    text = mock_root["c15"].read_text()
    assert not re.search(r"\.(dcm|png|jpe?g)\b|PatientName|PatientID|\b\d{3}-\d{2}-\d{4}\b", text)
    for line in sc.validate_c15_file(mock_root["c15"]):
        assert re.fullmatch(r"c_[0-9a-f]{8}", line["case_id"])


def test_c15_validator_rejects_missing_version_and_bad_analysed_line(mock_root):
    good = sc.validate_c15_file(mock_root["c15"])
    analysed = next(l for l in good if l["status"] == "ANALYSED")
    with pytest.raises(Exception):
        sc.validate_c15_line({k: v for k, v in analysed.items() if k != "git_sha"})
    with pytest.raises(Exception):
        sc.validate_c15_line({**analysed, "p_tb_cal": None})
    with pytest.raises(Exception):
        sc.validate_c15_line({**analysed, "patient_name": "x"})


def test_cli_entry_points(tmp_path, capsys):
    assert make_c10.main(["--n", "3", "--root", str(tmp_path)]) == 0
    assert make_c11.main(["--n", "3", "--root", str(tmp_path)]) == 0
    assert make_c15.main(["--n", "3", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "detections_tbx11k_development_v1.json" in out and "predictions.jsonl" in out
