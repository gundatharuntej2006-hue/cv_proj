"""P3-12: review case selection (pre-registered rule), overlays and the printable sheet.

P4's C12 (and its mock) are not in the repo yet, so a test-only C12 is built from fake C01 rows.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_det.data import DataPaths  # noqa: E402
from p3_det.data.fake_p1 import make_fake_p1_tree  # noqa: E402
from p3_xai import review  # noqa: E402


def fake_c12(c01: dict[str, dict], seed: int = 1, short_fn: bool = False) -> list[dict]:
    """C12-shaped development rows with every cell populated (FN made scarce if short_fn)."""
    r = np.random.default_rng(seed)
    rows = []
    for iid, lab in sorted(c01.items()):
        tb = int(lab["tb_label"])
        p_correct = 0.95 if (short_fn and tb == 1) else 0.6
        correct = r.random() < p_correct
        positive = (tb == 1) == correct
        p = float(r.uniform(0.84, 0.99) if positive else r.uniform(0.01, 0.17))
        cat = "TB" if positive and p >= 0.83 else ("REFER" if positive else "NOT_TB")
        rows.append({"image_id": iid, "split": "development", "model_id": "radino_crop_m0.10_clahe0_bin",
                     "p_tb_cal": p, "temperature": 1.3, "t_lower": 0.18, "t_upper": 0.83,
                     "category_pre_link": cat, "link_fired": False, "category_final": cat,
                     "screen_positive_nolink": positive, "screen_positive": positive,
                     "gate_pass": bool(r.random() > 0.03), "sanity_pass": True, "n_lesions": 0})
    return rows


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p3_12")
    root = make_fake_p1_tree(tmp / "p1", n_per_split=120)
    c01 = review.load_c01(DataPaths(root))
    return {"tmp": tmp, "root": root, "c01": c01, "rows": review.eligible(fake_c12(c01), c01)}


# --- the pre-registered rule ---------------------------------------------------------------------------

def test_doc_states_the_rule_the_code_uses():
    doc = (REPO / "docs" / "p3_gradcam_review.md").read_text(encoding="utf-8")
    assert f"default_rng({review.SEED})" in doc and "BEFORE any case was selected" in doc
    for cell in ("true positive", "true negative", "false negative", "false positive"):
        assert cell in doc
    assert "screen_positive == (tb_label == 1)" in doc and "gate_pass = true" in doc
    assert "not** proof of clinical correctness" in doc


def test_selection_40_cases_10_per_cell(data):
    sel = review.select(data["rows"])
    assert len(sel) == 40 and len({s["image_id"] for s in sel}) == 40
    cells = {n: [s for s in sel if s["cell"] == n] for n, _, _ in review.CELLS}
    assert {n: len(v) for n, v in cells.items()} == {"TP": 10, "TN": 10, "FN": 10, "FP": 10}
    for name, tb, ok in review.CELLS:
        assert all(s["tb_label"] == tb and s["correct"] is ok for s in cells[name])
    assert sum(s["correct"] for s in sel) == 20
    assert [s["review_no"] for s in sel] == [f"R{k:02d}" for k in range(1, 41)]


def test_correct_means_screen_positive_matches_label(data):
    for s in review.select(data["rows"]):
        assert s["correct"] == (s["screen_positive"] == (s["tb_label"] == 1))
        assert s["correct"] == ((s["category_final"] in ("TB", "REFER")) == (s["tb_label"] == 1))


def test_only_eligible_rows(data):
    c12 = fake_c12(data["c01"])
    rejected = {r["image_id"] for r in c12 if not r["gate_pass"]}
    assert rejected and not rejected & {s["image_id"] for s in review.select(review.eligible(c12, data["c01"]))}
    with pytest.raises(ValueError, match="development only"):
        review.eligible([{**c12[0], "split": "internal_test"}], data["c01"])


def test_deterministic_seeded_and_order_shuffled(data):
    a, b = review.select(data["rows"]), review.select(data["rows"])
    assert a == b
    assert review.select(data["rows"], seed=1) != a
    cells_in_order = [s["cell"] for s in a]
    assert cells_in_order != sorted(cells_in_order, key=["TP", "TN", "FN", "FP"].index)   # not grouped


def test_shortfall_filled_from_same_group_and_recorded(data):
    rows = review.eligible(fake_c12(data["c01"], short_fn=True), data["c01"])
    n_fn = sum(r["tb_label"] == 1 and not r["correct"] for r in rows)
    assert 0 < n_fn < 10
    sel = review.select(rows)
    assert len(sel) == 40 and sum(s["correct"] for s in sel) == 20
    filled = [s for s in sel if s["filled_from"]]
    assert len(filled) == 10 - n_fn and all(s["cell"] == "FP" and s["filled_from"] == "filled for FN" for s in filled)


def test_selection_does_not_look_at_heatmaps():
    import inspect
    assert set(inspect.signature(review.select).parameters) == {"rows", "seed", "per_cell"}


# --- overlays + sheet ---------------------------------------------------------------------------------

def test_render_40_overlays_and_sheet(data, monkeypatch):
    monkeypatch.setenv("TB_MOCK", "1")                        # mock explain()/detect(): no RAD-DINO/D-FINE needed
    out = data["tmp"] / "review"
    sel = review.select(data["rows"])
    p = review.write_selection(sel, out)
    sel2 = review.read_selection(p)
    assert [s["image_id"] for s in sel2] == [s["image_id"] for s in sel]
    sheet = review.render(sel2, paths=DataPaths(data["root"]), out_dir=out, decisions_dir=data["tmp"] / "nodec")
    pngs = sorted(out.glob("R*.png"))
    assert len(pngs) == 40
    html_text = sheet.read_text(encoding="utf-8")
    assert html_text.count('<section class="case">') == 40
    assert html_text.count('<div class="comment"></div>') == 40           # empty guide-comment boxes
    assert html_text.count("attention evidence, not a finding") >= 2      # caveat top and bottom
    for s in sel2:
        assert f"case {s['image_id']}" in html_text and f"{s['p_tb_cal']:.3f}" in html_text
        assert (out / s["overlay"]).exists() and f'src="{s["overlay"]}"' in html_text
    # only case ids, no other identifiers: every file name is R##_<image_id>.png
    assert all(re.fullmatch(r"R\d\d_[A-Za-z0-9_-]+\.png", p.name) for p in pngs)


def test_c10_boxes_above_threshold_only(tmp_path):
    import json
    p = tmp_path / "detections_tbx11k_development_v1.json"
    p.write_text(json.dumps({"conf_threshold": 0.4, "images": [
        {"image_id": "tb0001", "boxes": [[1, 1, 20, 20, 0.9], [30, 30, 50, 50, 0.1]]}]}))
    f = review.boxes_from_c10(p)
    assert f("tb0001") == [{"xyxy": [1, 1, 20, 20], "score": 0.9}] and f("h0002") == []
