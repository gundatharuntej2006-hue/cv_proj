"""P3-05: detection metrics: AP50 vs pycocotools, hand-computed FROC, edge cases, CIs."""
from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from p3_det.metrics import (  # noqa: E402
    ImageGT, MetricResult, ap50, bootstrap_ci, detections_from_c10, froc, froc_curve,
    ground_truth_from_coco, image_level_lesion_sensitivity, match_image, subtype_stratified, wilson_ci)


def box(x, y, w=40, h=40):
    return [x, y, x + w, y + h]


def det(b, s):
    return [*b, s]


# --- Fixture 1 (also the hand-computed FROC fixture) ----------------------------------------------
# 4 images, 3 lesions. L1, L2 on image A; L3 on image B; C and D are non-TB.
L1, L2, L3 = box(100, 100), box(300, 300), box(200, 50)
F1 = {
    "gt": {"A": ImageGT([L1, L2], "active"), "B": ImageGT([L3], "latent"), "C": ImageGT(), "D": ImageGT()},
    "det": {
        "A": [det(L1, 0.9),                 # d1 TP (finds L1)
              det(box(102, 101), 0.6),      # d2 duplicate on L1 -> FP (L1 already taken)
              det(L2, 0.15)],               # d3 TP (finds L2)
        "B": [det(box(400, 400), 0.8),      # d4 FP (nothing there)
              det(box(203, 52), 0.5)],      # d5 TP (finds L3, IoU = 37*38/(2*1600-37*38) = 0.78)
        "C": [det(box(50, 400), 0.7),       # d6 FP
              det(box(10, 10), 0.2)],       # d7 FP
        "D": [det(box(250, 250), 0.4)],     # d8 FP
    },
}
# Worked FROC (thresholds high -> low; FP/image = FP / 4 images; sensitivity = TP / 3 lesions):
#   score  kind  cumFP  FP/img  cumTP  sens
#   0.90   TP      0    0.00     1     1/3
#   0.80   FP      1    0.25     1     1/3
#   0.70   FP      2    0.50     1     1/3
#   0.60   FP(dup) 3    0.75     1     1/3
#   0.50   TP      3    0.75     2     2/3
#   0.40   FP      4    1.00     2     2/3
#   0.20   FP      5    1.25     2     2/3
#   0.15   TP      5    1.25     3     3/3
# Sensitivity at k FP/image = best sensitivity with FP/image <= k:
#   k = 0.5 -> 1/3 ;  k = 1 -> 2/3 (at 0.40) ;  k = 2 -> 1 (at 0.15) ;  k = 4 -> 1 (curve ends at 1.25)

# --- Fixture 2: overlapping lesions, tied scores, an image with 120 detections (max-dets cap) ------
F2 = {
    "gt": {"a": ImageGT([box(100, 100, 60, 60), box(130, 100, 60, 60)], "active"),
           "b": ImageGT([box(300, 300, 30, 30)], "active+latent"),
           "c": ImageGT(), "d": ImageGT([box(10, 10, 100, 100)], "active")},
    "det": {
        "a": [det(box(115, 100, 60, 60), 0.7),   # IoU 0.6 with BOTH lesions; COCO tie rule -> the later one
              det(box(100, 100, 60, 60), 0.7),   # tie in score; takes the first lesion (IoU 1)
              det(box(131, 101, 60, 60), 0.3)],  # both lesions taken -> FP
        "b": [det(box(307, 307, 30, 30), 0.7)],  # IoU 23*23/(900+900-529) = 0.42 < 0.5 -> FP
        "c": [det(box(5 * k % 400, 7 * k % 400, 20, 20), 0.05 + 0.001 * k) for k in range(120)],
        "d": [det(box(12, 12, 98, 99), 0.06)],   # TP with a low score, ranked among c's clutter
    },
}


def _random_fixture(seed: int, n: int = 40):
    r = np.random.default_rng(seed)
    gt, dets = {}, {}
    for i in range(n):
        iid = f"img{i:03d}"
        k = int(r.integers(0, 4)) if i % 3 else 0
        boxes = []
        for _ in range(k):
            x, y = r.uniform(0, 440, 2)
            w, h = r.uniform(15, 70, 2)
            boxes.append([x, y, x + w, y + h])
        gt[iid] = ImageGT(boxes, ["active", "latent", "active+latent"][(i // 3) % 3] if boxes else None)
        d = []
        for b in boxes:
            if r.random() < 0.7:
                j = r.normal(0, 6, 4)
                d.append([*(np.array(b) + j), float(r.uniform(0.2, 1))])
        for _ in range(int(r.integers(0, 4))):
            x, y = r.uniform(0, 440, 2)
            d.append([x, y, x + 40, y + 40, float(r.uniform(0.01, 0.8))])
        dets[iid] = np.array([[min(b[0], b[2] - 1), min(b[1], b[3] - 1), *b[2:]] for b in d]).reshape(-1, 5)
    return {"gt": gt, "det": dets}


FIXTURES = {"froc_fixture": F1, "overlap_ties_maxdets": F2, "random40": _random_fixture(7)}


def _pycocotools_ap50(fx) -> float:
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids = sorted(fx["gt"])                                    # same image order as ours
    num = {iid: k + 1 for k, iid in enumerate(ids)}
    images, anns, dets = [], [], []
    for iid in ids:
        images.append({"id": num[iid], "width": 512, "height": 512})
        for x0, y0, x1, y1 in fx["gt"][iid].boxes:
            anns.append({"id": len(anns) + 1, "image_id": num[iid], "category_id": 1, "iscrowd": 0,
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0)})
        for x0, y0, x1, y1, s in np.asarray(fx["det"].get(iid, []), float).reshape(-1, 5):
            dets.append({"image_id": num[iid], "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "tb_lesion"}]}
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[1])                                 # AP @ IoU 0.50, all areas, maxDets 100


# --- AP50 vs pycocotools -------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(FIXTURES))
def test_ap50_matches_pycocotools(name):
    fx = FIXTURES[name]
    ours = ap50(fx["det"], fx["gt"], ci=False)
    ref = _pycocotools_ap50(fx)
    assert abs(ours.value - ref) < 1e-3, (name, ours.value, ref)


def test_ap50_result_has_counts_and_ci():
    r = ap50(F1["det"], F1["gt"], B=300, seed=1)
    assert (r.n_images, r.n_tb_images, r.n_non_tb_images, r.n_boxes) == (4, 2, 2, 3)
    assert r.ci_method == "bootstrap_percentile" and r.ci_low <= r.value <= r.ci_high
    assert r == ap50(F1["det"], F1["gt"], B=300, seed=1)      # fixed seed -> identical CI


def test_max_dets_cap_only_affects_ap():
    # image c has 120 detections; COCO keeps the top 100 for AP. FROC keeps all.
    fp, _ = froc_curve(F2["det"], F2["gt"])
    assert fp.max() * 4 == 120 + 1 + 1                        # all 122 FPs counted (incl. b's miss, a's extra)


# --- FROC ----------------------------------------------------------------------------------------

def test_froc_hand_computed():
    r = {x.name: x for x in froc(F1["det"], F1["gt"], fp_per_image=[0.5, 1, 2, 4], ci=False)}
    assert r["froc_sens_at_0.5fp"].value == pytest.approx(1 / 3)
    assert r["froc_sens_at_1fp"].value == pytest.approx(2 / 3)
    assert r["froc_sens_at_2fp"].value == pytest.approx(1.0)
    assert r["froc_sens_at_4fp"].value == pytest.approx(1.0)
    assert "curve ends at 1.250 FP/image < 4" in r["froc_sens_at_4fp"].note
    fp, sens = froc_curve(F1["det"], F1["gt"])
    assert fp.tolist() == [0, 0.25, 0.5, 0.75, 0.75, 1.0, 1.25, 1.25]
    assert np.allclose(sens, [1 / 3, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1])
    assert all(x.n_boxes == 3 and x.n_images == 4 for x in r.values())


def test_froc_ci():
    for x in froc(F1["det"], F1["gt"], B=300, seed=2):
        assert x.ci_method == "bootstrap_percentile" and 0 <= x.ci_low <= x.ci_high <= 1


# --- image-level sensitivity ---------------------------------------------------------------------

def test_image_level_lesion_sensitivity():
    # A: d1 (0.9) hits L1; B: d5 (0.5) hits L3 -> both TB images hit at threshold 0.5
    r = image_level_lesion_sensitivity(F1["det"], F1["gt"], conf_threshold=0.5)
    assert r.value == 1.0 and r.n_tb_images == 2 and r.ci_method == "wilson"
    # at 0.55 only A is hit (B's TP has score 0.5)
    assert image_level_lesion_sensitivity(F1["det"], F1["gt"], 0.55).value == 0.5
    assert image_level_lesion_sensitivity(F1["det"], F1["gt"], 0.95).value == 0.0


# --- edge cases ----------------------------------------------------------------------------------

def test_no_detections():
    r = ap50({}, F1["gt"], B=50)
    assert r.value == 0.0 and r.n_boxes == 3
    assert all(x.value == 0.0 for x in froc({}, F1["gt"], ci=False))
    assert image_level_lesion_sensitivity({}, F1["gt"], 0.5).value == 0.0


def test_no_ground_truth_boxes():
    gt = {"C": ImageGT(), "D": ImageGT()}
    dets = {"C": [det(box(1, 1), 0.9)]}
    r = ap50(dets, gt)
    assert math.isnan(r.value) and r.ci_method == "none" and "undefined" in r.note and r.n_boxes == 0
    assert all(math.isnan(x.value) for x in froc(dets, gt))
    with pytest.raises(ValueError, match="TB image"):
        image_level_lesion_sensitivity(dets, gt, 0.5)
    with pytest.raises(ValueError, match="empty"):
        ap50({}, {})


def test_duplicate_detections_count_once():
    gt = {"x": ImageGT([L1], "active")}
    dets = {"x": [det(L1, 0.9), det(box(101, 101), 0.8), det(box(99, 100), 0.7)]}
    m = match_image(np.array(dets["x"]), gt["x"].boxes)
    assert m.tp.tolist() == [True, False, False]              # one TP, two FPs
    assert ap50(dets, gt, ci=False).value == pytest.approx(1.0)   # TP ranked first
    fp, sens = froc_curve(dets, gt)
    assert fp.tolist() == [0, 1, 2] and sens.tolist() == [1, 1, 1]


def test_detections_for_unknown_image_rejected():
    with pytest.raises(ValueError, match="not in ground_truth"):
        ap50({"ghost": [det(L1, 0.9)]}, F1["gt"])


def test_metric_result_refuses_missing_n():
    base = dict(name="x", value=0.5, ci_low=0.4, ci_high=0.6, n_images=10, n_tb_images=4,
                n_non_tb_images=6, n_boxes=5, ci_method="wilson")
    MetricResult(**base)
    for bad in ({"n_images": None}, {"n_boxes": None}, {"n_images": 0, "n_tb_images": 0, "n_non_tb_images": 0},
                {"n_tb_images": 5}, {"n_images": 2.5}, {"ci_method": "made_up"}):
        with pytest.raises(ValueError):
            MetricResult(**{**base, **bad})


# --- subtype stratification ----------------------------------------------------------------------

def test_subtype_stratified():
    fx = FIXTURES["random40"]
    out = subtype_stratified(fx["det"], fx["gt"], conf_threshold=0.5, B=100)
    assert set(out) == {"active", "latent", "active+latent"}
    n_non_tb = sum(not g.is_tb for g in fx["gt"].values())
    for st, results in out.items():
        n_tb = sum(g.subtype == st for g in fx["gt"].values())
        assert [r.name for r in results][:2] == ["ap50", "image_level_lesion_sensitivity"]
        assert all(r.n_tb_images == n_tb and r.n_non_tb_images == n_non_tb for r in results)
        assert all(f"subtype={st}" in r.note and "small n" in r.note for r in results)
    assert all("not reliable (very small n)" in r.note for r in out["latent"])
    assert not any("not reliable" in r.note for r in out["active"])


# --- CIs -----------------------------------------------------------------------------------------

def test_wilson_known_value():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson_ci(0, 5)[0] == 0.0 and wilson_ci(5, 5)[1] == 1.0


def test_bootstrap_resamples_images_and_keeps_strata():
    strata = np.array(["tb"] * 3 + ["non_tb"] * 7)
    seen = []

    def stat(idx):
        seen.append(idx)
        return float((strata[idx] == "tb").mean())
    lo, hi, dropped = bootstrap_ci(10, stat, strata=strata, B=200, seed=3)
    assert (lo, hi, dropped) == (0.3, 0.3, 0)                  # stratum sizes fixed in every replicate
    assert all(len(i) == 10 for i in seen)
    assert bootstrap_ci(10, lambda i: float(i.mean()), B=200, seed=3) == bootstrap_ci(
        10, lambda i: float(i.mean()), B=200, seed=3)


# --- loaders -------------------------------------------------------------------------------------

def test_loaders_from_contract_shapes():
    c10 = {"coords": "full512", "images": [{"image_id": "A", "boxes": [det(L1, 0.9)]}, {"image_id": "C", "boxes": []}]}
    d = detections_from_c10(c10)
    assert d["A"].shape == (1, 5) and d["C"].shape == (0, 5)
    with pytest.raises(ValueError):
        detections_from_c10({**c10, "coords": "crop"})
    coco = {"images": [{"id": 1, "file_name": "A"}, {"id": 2, "file_name": "C"}],
            "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1], "orig_bbox_512": [100, 100, 40, 40],
                             "tb_subtype": "active"},
                            {"id": 2, "image_id": 1, "bbox": [0, 0, 1, 1], "orig_bbox_512": [300, 300, 40, 40],
                             "tb_subtype": "latent"}]}
    gt = ground_truth_from_coco(coco, image_ids=["A", "C", "Z"])
    assert gt["A"].boxes.tolist() == [L1, L2] and gt["A"].subtype == "active+latent"
    assert gt["C"].subtype is None and not gt["Z"].is_tb
    assert ap50(d, gt, ci=False).n_images == 3
