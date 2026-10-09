"""P3-05: lesion-detection metrics (methodology 12.4), imported by P4.

All boxes are [x0, y0, x1, y1] in FULL 512-image coordinates, and ground truth is ALL original
annotated boxes of each image, including any outside the lung crop (D21). A detector that only
looks inside the crop is therefore (correctly) penalised for lesions it could not see.

Inputs
------
ground_truth : {image_id: ImageGT}   EVERY evaluated image, TB and non-TB (non-TB: no boxes).
               This defines the image set; images without detections simply have none.
detections   : {image_id: array [m, 5]}  rows [x0, y0, x1, y1, score], e.g. C10 ``boxes``.
               Missing ids mean "no detections"; ids not in ground_truth are an error.

Matching (shared by all metrics, identical to pycocotools)
--------
Per image, detections are taken in descending score order. Each one is matched to the
not-yet-matched ground-truth box it overlaps most, if that IoU is at least 0.50. A matched
detection is a true positive (TP). Every other detection is a false positive (FP), including a
second detection of a lesion that was already found (duplicates count once).

Results
-------
Each metric returns a MetricResult with the value, a 95% CI and the counts it rests on
(n_images, n_tb_images, n_non_tb_images, n_boxes). A result cannot be created without them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .stats import DEFAULT_B, DEFAULT_SEED, bootstrap_ci, wilson_ci

IOU_THR = 0.5
COCO_MAX_DETS = 100
SUBTYPES = ("active", "latent", "active+latent")
SMALL_N = 30            # strata with fewer TB images than this are flagged as unreliable
LATENT_NOTE = "not reliable (very small n)"


# --- data types ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageGT:
    """Ground truth for one image: all annotated lesion boxes (512 coords) and its TB subtype.

    subtype is "active", "latent", "active+latent" for TB images and None for non-TB images.
    """
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    subtype: str | None = None

    def __post_init__(self):
        b = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        object.__setattr__(self, "boxes", b)
        if self.subtype is not None and self.subtype not in SUBTYPES:
            raise ValueError(f"subtype must be one of {SUBTYPES} or None, got {self.subtype!r}")
        if len(b) and ((b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])).any():
            raise ValueError("ground-truth boxes must have positive width and height")

    @property
    def is_tb(self) -> bool:
        return len(self.boxes) > 0


@dataclass(frozen=True)
class MetricResult:
    """One reported number with its 95% CI and the sample it rests on.

    ci_method: "bootstrap_percentile" (images resampled), "wilson", or "none" (CI not requested).
    value is NaN only when the metric is undefined (e.g. AP with no ground-truth boxes); the
    note then says why.
    """
    name: str
    value: float
    ci_low: float
    ci_high: float
    n_images: int
    n_tb_images: int
    n_non_tb_images: int
    n_boxes: int
    ci_method: str
    note: str = ""

    def __post_init__(self):
        for k in ("n_images", "n_tb_images", "n_non_tb_images", "n_boxes"):
            v = getattr(self, k)
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0:
                raise ValueError(f"MetricResult needs a non-negative integer {k}, got {v!r}")
        if self.n_images == 0:
            raise ValueError("refusing to report a metric computed on 0 images")
        if self.n_tb_images + self.n_non_tb_images != self.n_images:
            raise ValueError("n_tb_images + n_non_tb_images must equal n_images")
        if self.ci_method not in ("bootstrap_percentile", "wilson", "none"):
            raise ValueError(f"unknown ci_method {self.ci_method!r}")

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --- geometry and matching -----------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a [n,4] and b [m,4] (xyxy, no +1 pixel convention, as COCO)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


@dataclass(frozen=True)
class _Matched:
    scores: np.ndarray      # [m] detection scores, descending
    tp: np.ndarray          # [m] bool
    gt_hit: np.ndarray      # [g] bool: lesion found by some detection (at any score)
    gt_hit_score: np.ndarray  # [g] score of the detection that found it (nan if never)
    n_gt: int


def match_image(dets: np.ndarray, gt: np.ndarray, iou_thr: float = IOU_THR,
                max_dets: int | None = None) -> _Matched:
    """Greedy, score-ordered one-to-one matching (pycocotools semantics)."""
    dets = np.asarray(dets, float).reshape(-1, 5)
    order = np.argsort(-dets[:, 4], kind="mergesort")
    if max_dets is not None:
        order = order[:max_dets]
    dets = dets[order]
    ious = iou_matrix(dets[:, :4], gt)
    gt_taken = np.zeros(len(gt), bool)
    hit_score = np.full(len(gt), np.nan)
    tp = np.zeros(len(dets), bool)
    for d in range(len(dets)):
        best, m = min(iou_thr, 1 - 1e-10), -1
        for g in range(len(gt)):
            if gt_taken[g] or ious[d, g] < best:
                continue
            best, m = ious[d, g], g
        if m >= 0:
            gt_taken[m] = True
            hit_score[m] = dets[d, 4]
            tp[d] = True
    return _Matched(dets[:, 4].copy(), tp, gt_taken, hit_score, len(gt))


class _Prepared:
    """Per-image matching done once; metrics on any image subset (bootstrap) reuse it."""

    def __init__(self, detections: Mapping[str, np.ndarray], ground_truth: Mapping[str, ImageGT],
                 max_dets: int | None):
        unknown = set(detections) - set(ground_truth)
        if unknown:
            raise ValueError(f"detections for images not in ground_truth: {sorted(unknown)[:5]}")
        if not ground_truth:
            raise ValueError("ground_truth is empty: no images to evaluate")
        self.ids = sorted(ground_truth)                       # fixed order (as COCO's sorted imgIds)
        self.gt = [ground_truth[i] for i in self.ids]
        self.dets = [np.asarray(detections.get(i, np.zeros((0, 5))), float).reshape(-1, 5) for i in self.ids]
        self.matched = [match_image(d, g.boxes, max_dets=max_dets) for d, g in zip(self.dets, self.gt)]
        self.is_tb = np.array([g.is_tb for g in self.gt])
        self.strata = np.where(self.is_tb, "tb", "non_tb")

    def counts(self, idx: np.ndarray) -> dict:
        tb = int(self.is_tb[idx].sum())
        return {"n_images": len(idx), "n_tb_images": tb, "n_non_tb_images": len(idx) - tb,
                "n_boxes": int(sum(self.matched[i].n_gt for i in idx))}


# --- AP@0.50 -------------------------------------------------------------------------------------

def _ap_from(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP from pooled detections."""
    if n_gt == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order]
    tps, fps = np.cumsum(tp), np.cumsum(~tp)
    recall = tps / n_gt
    precision = tps / np.maximum(tps + fps, np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    q = np.zeros(101)                                   # precision at recall 0, 0.01, ..., 1
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    valid = idx < len(precision)
    q[valid] = precision[idx[valid]]
    return float(q.mean())


def _ap_subset(p: _Prepared, idx: np.ndarray) -> float:
    ms = [p.matched[i] for i in idx]
    scores = np.concatenate([m.scores for m in ms]) if ms else np.zeros(0)
    tp = np.concatenate([m.tp for m in ms]) if ms else np.zeros(0, bool)
    return _ap_from(scores, tp, sum(m.n_gt for m in ms))


def ap50(detections: Mapping[str, np.ndarray], ground_truth: Mapping[str, ImageGT], *, ci: bool = True,
         B: int = DEFAULT_B, seed: int = DEFAULT_SEED, max_dets: int = COCO_MAX_DETS) -> MetricResult:
    """Average Precision at IoU 0.50 for the single TB-lesion class.

    Plain language: rank every predicted box in the evaluation set by confidence. Walking down
    the list, precision is the share of boxes so far that hit a real lesion and recall is the
    share of real lesions found so far. AP summarises that precision-recall curve in one number
    between 0 and 1 (higher is better), exactly as pycocotools' AP50: at most 100 boxes per image,
    precision made non-increasing, averaged over 101 recall levels 0, 0.01, ..., 1.
    CI: percentile bootstrap over images, stratified TB / non-TB. NaN if there are no GT boxes.
    """
    p = _Prepared(detections, ground_truth, max_dets)
    idx = np.arange(len(p.ids))
    value = _ap_subset(p, idx)
    return _result("ap50", value, p, idx, ci, B, seed, lambda s: _ap_subset(p, s),
                   note="undefined: no ground-truth boxes" if math.isnan(value) else "")


def ap50_difference(detections_a: Mapping[str, np.ndarray], detections_b: Mapping[str, np.ndarray],
                    ground_truth: Mapping[str, ImageGT], *, B: int = DEFAULT_B, seed: int = DEFAULT_SEED,
                    max_dets: int = COCO_MAX_DETS) -> MetricResult:
    """AP50(b) - AP50(a) on the same images, with a PAIRED bootstrap CI.

    Plain language: how much better (positive) or worse (negative) detector b is than detector a.
    Every bootstrap replicate resamples one set of images and scores both detectors on it, so
    the CI reflects the difference itself, not two independent uncertainties.
    """
    pa = _Prepared(detections_a, ground_truth, max_dets)
    pb = _Prepared(detections_b, ground_truth, max_dets)
    idx = np.arange(len(pa.ids))
    value = _ap_subset(pb, idx) - _ap_subset(pa, idx)
    return _result("ap50_difference", value, pa, idx, True, B, seed,
                   lambda s: _ap_subset(pb, s) - _ap_subset(pa, s),
                   note="undefined: no ground-truth boxes" if math.isnan(value) else "")


# --- image-level lesion sensitivity --------------------------------------------------------------

def _image_hit(dets: np.ndarray, gt: ImageGT, conf_threshold: float) -> bool:
    d = dets[dets[:, 4] >= conf_threshold]
    return bool(len(d) and (iou_matrix(d[:, :4], gt.boxes) >= IOU_THR).any())


def image_level_lesion_sensitivity(detections: Mapping[str, np.ndarray], ground_truth: Mapping[str, ImageGT],
                                   conf_threshold: float) -> MetricResult:
    """Share of TB images with at least one predicted box (score >= conf_threshold) overlapping an
    annotated box at IoU >= 0.50 (methodology 12.4).

    Plain language: "of the TB X-rays, on how many did the detector put a box on a real lesion?"
    Only TB images (images with annotated boxes) count; non-TB images are reported in the n but
    do not affect the value. CI: Wilson (a proportion over images).
    """
    p = _Prepared(detections, ground_truth, None)
    tb = np.flatnonzero(p.is_tb)
    if not len(tb):
        raise ValueError("image-level lesion sensitivity needs at least one TB image with boxes")
    hits = sum(_image_hit(p.dets[i], p.gt[i], conf_threshold) for i in tb)
    lo, hi = wilson_ci(hits, len(tb))
    return MetricResult("image_level_lesion_sensitivity", hits / len(tb), lo, hi,
                        **p.counts(np.arange(len(p.ids))), ci_method="wilson",
                        note=f"{hits}/{len(tb)} TB images hit at conf >= {conf_threshold}")


# --- FROC ----------------------------------------------------------------------------------------

def _froc_points(p: _Prepared, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fp_per_image, lesion_sensitivity) at every distinct score threshold, high to low."""
    ms = [p.matched[i] for i in idx]
    n_gt = sum(m.n_gt for m in ms)
    scores = np.concatenate([m.scores for m in ms]) if ms else np.zeros(0)
    tp = np.concatenate([m.tp for m in ms]) if ms else np.zeros(0, bool)
    if not len(scores):
        return np.zeros(0), np.zeros(0)
    # With one-to-one matching, lesions found at threshold t == TP detections with score >= t.
    order =np.argsort(-scores, kind="mergesort")
    s, t = scores[order], tp[order]
    ends = np.flatnonzero(np.append(s[1:] != s[:-1], True))      # last row of each tied score
    fp_at = np.cumsum(~t)[ends] / len(idx)
    sens_at = np.cumsum(t)[ends] / n_gt if n_gt else np.full(len(ends), np.nan)
    return fp_at, sens_at


def _froc_value(p: _Prepared, idx: np.ndarray, rate: float) -> float:
    ms_gt = sum(p.matched[i].n_gt for i in idx)
    if ms_gt == 0:
        return math.nan
    fp, sens = _froc_points(p, idx)
    ok = fp <= rate
    return float(sens[ok].max()) if ok.any() else 0.0


def froc(detections: Mapping[str, np.ndarray], ground_truth: Mapping[str, ImageGT],
         fp_per_image: Sequence[float] = (1, 2, 4), *, ci: bool = True, B: int = DEFAULT_B,
         seed: int = DEFAULT_SEED) -> list[MetricResult]:
    """Free-response ROC: lesion sensitivity at 1, 2 and 4 false positives per image.

    Plain language: lower the confidence threshold step by step. At each step count the share of
    all annotated lesions that have been found (lesion sensitivity) and the average number of
    wrong boxes per X-ray over ALL evaluated images, TB and non-TB (FPs per image). The value at
    "k FP/image" is the best lesion sensitivity reachable while averaging at most k wrong boxes
    per image (step function, no interpolation). If even the lowest threshold stays below k
    FP/image, the note says so: the value is then the sensitivity at the lowest threshold.
    All detections are used (no 100-per-image cap). CI: bootstrap over images, stratified.
    """
    p = _Prepared(detections, ground_truth, None)
    idx = np.arange(len(p.ids))
    fp_all, _ = _froc_points(p, idx)
    max_fp = float(fp_all.max()) if len(fp_all) else 0.0
    out = []
    for rate in fp_per_image:
        value = _froc_value(p, idx, rate)
        note = "undefined: no ground-truth boxes" if math.isnan(value) else (
            f"curve ends at {max_fp:.3f} FP/image < {rate}" if max_fp < rate else "")
        out.append(_result(f"froc_sens_at_{rate:g}fp", value, p, idx, ci, B, seed,
                           lambda s, r=rate: _froc_value(p, s, r), note=note))
    return out


def froc_curve(detections: Mapping[str, np.ndarray], ground_truth: Mapping[str, ImageGT]) -> tuple[np.ndarray, np.ndarray]:
    """Full FROC curve (fp_per_image, lesion_sensitivity) for plotting; thresholds high to low."""
    p = _Prepared(detections, ground_truth, None)
    return _froc_points(p, np.arange(len(p.ids)))


# --- subtype stratification ----------------------------------------------------------------------

def subtype_stratified(detections: Mapping[str, np.ndarray], ground_truth: Mapping[str, ImageGT],
                       conf_threshold: float, fp_per_image: Sequence[float] = (1, 2, 4), *,
                       ci: bool = True, B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> dict[str, list[MetricResult]]:
    """The metrics above, separately for active, latent and active+latent TB (methodology 12.4).

    Each stratum = the TB images of that subtype PLUS all non-TB images, so false positives on
    healthy and sick-non-TB X-rays still count against AP and FROC. Image-level sensitivity uses
    the stratum's TB images only. Latent TB is always flagged "not reliable (very small n)"; any
    stratum with fewer than 30 TB images is flagged as small. Strata with no images are omitted.
    """
    non_tb = {i: g for i, g in ground_truth.items() if not g.is_tb}
    out: dict[str, list[MetricResult]] = {}
    for st in SUBTYPES:
        tb = {i: g for i, g in ground_truth.items() if g.is_tb and g.subtype == st}
        if not tb:
            continue
        gts = {**tb, **non_tb}
        dets = {i: d for i, d in detections.items() if i in gts}
        flags = []
        if st == "latent":
            flags.append(LATENT_NOTE)
        if len(tb) < SMALL_N:
            flags.append(f"small n ({len(tb)} TB images < {SMALL_N})")
        res = [ap50(dets, gts, ci=ci, B=B, seed=seed),
               image_level_lesion_sensitivity(dets, gts, conf_threshold),
               *froc(dets, gts, fp_per_image, ci=ci, B=B, seed=seed)]
        out[st] = [_with_note(r, f"subtype={st}", *flags) for r in res]
    return out


# --- helpers -------------------------------------------------------------------------------------

def _with_note(r: MetricResult, *notes: str) -> MetricResult:
    note = "; ".join(n for n in (r.note, *notes) if n)
    return MetricResult(**{**r.as_dict(), "note": note})


def _result(name, value, p: _Prepared, idx, ci, B, seed, stat, note="") -> MetricResult:
    lo = hi = math.nan
    method = "none"
    if ci and not math.isnan(value):
        lo, hi, dropped = bootstrap_ci(len(idx), stat, strata=p.strata[idx], B=B, seed=seed)
        method = "bootstrap_percentile"
        if dropped:
            note = "; ".join(x for x in (note, f"{dropped}/{B} bootstrap replicates undefined") if x)
    return MetricResult(name, float(value), lo, hi, **p.counts(idx), ci_method=method, note=note)


# --- loaders from the contract files --------------------------------------------------------------

def detections_from_c10(doc: Mapping) -> dict[str, np.ndarray]:
    """C10 detections JSON (already parsed) -> {image_id: [m,5]}; checks coords == full512."""
    if doc.get("coords") != "full512":
        raise ValueError("C10 boxes must be in full512 coordinates")
    return {im["image_id"]: np.asarray(im["boxes"], float).reshape(-1, 5) for im in doc["images"]}


def ground_truth_from_coco(doc: Mapping, image_ids: Iterable[str] | None = None,
                           subtype_of: Mapping[str, str | None] | None = None) -> dict[str, ImageGT]:
    """C06-style COCO (parsed) -> {image_id: ImageGT} using ``orig_bbox_512`` (full 512 coords).

    image_ids: the evaluated image set (e.g. all C01 ids of the split); defaults to every COCO
    image. subtype_of: image-level subtype (from C01 category); else taken from the boxes'
    ``tb_subtype`` (one subtype per image, "active+latent" if both occur).

    NOTE: C06 drops boxes lying fully outside the lung crop, so it cannot supply "all original
    boxes" (D21) for such images. See docs/p3_contract_notes.md item 11.
    """
    name = {im["id"]: im["file_name"] for im in doc["images"]}
    boxes: dict[str, list] = {n: [] for n in name.values()}
    subs: dict[str, set] = {n: set() for n in name.values()}
    for a in doc["annotations"]:
        n = name[a["image_id"]]
        x, y, w, h = a.get("orig_bbox_512", a["bbox"])
        boxes[n].append([x, y, x + w, y + h])
        if a.get("tb_subtype"):
            subs[n].add(a["tb_subtype"])
    ids = list(image_ids) if image_ids is not None else list(name.values())
    out = {}
    for i in ids:
        b = np.asarray(boxes.get(i, []), float).reshape(-1, 4)
        if subtype_of is not None:
            st = subtype_of.get(i)
        else:
            s = subs.get(i, set())
            st = "active+latent" if s >= {"active", "latent"} or "active+latent" in s else (next(iter(s)) if s else None)
        out[i] = ImageGT(b, st if len(b) else None)
    return out
