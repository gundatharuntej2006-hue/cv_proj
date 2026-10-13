# -*- coding: utf-8 -*-
"""P2-05 acceptance: each named degradation rejected with the correct reason, and clean
radiographs left alone. Both halves matter — a rule that rejects everything passes the
first criterion and fails the project."""
import numpy as np
import pytest

from p2_gate import quality as Q


def synthetic_cxr(seed=0, h=512, w=512):
    """A crude chest-like image: bright lung-ish field, darker mediastinum, noise.
    Not a radiograph, but it has the property that matters here — a realistic spread of
    intensities rather than a flat field."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = 0.30 + 0.35 * np.exp(-((x - w * 0.30) ** 2 + (y - h * 0.45) ** 2) / (2 * (w * 0.18) ** 2))
    a += 0.35 * np.exp(-((x - w * 0.70) ** 2 + (y - h * 0.45) ** 2) / (2 * (w * 0.18) ** 2))
    a += 0.10 * rng.normal(size=(h, w))
    return np.clip(a, 0, 1).astype(np.float32)


IMAGES = [synthetic_cxr(s) for s in range(12)]


# ------------------------------------------------------------------ rejections
@pytest.mark.parametrize("kind,expected", [
    ("black", "blank"),
    ("white", "blank"),
    ("under_exposed", "under_exposed"),
    ("over_exposed", "over_exposed"),
    ("heavily_cropped", "heavily_cropped"),
])
def test_each_named_degradation_is_rejected_with_the_right_reason(kind, expected):
    for im in IMAGES[:6]:
        r = Q.assess(Q.degrade(im, kind, severity=1.0))
        assert not r.passed, "%s at full severity was accepted" % kind
        assert expected in r.reasons, "%s gave %s, expected %s" % (kind, r.reasons, expected)


def test_a_sixty_percent_crop_is_rejected():
    """The work plan names this case explicitly."""
    for im in IMAGES[:6]:
        narrow = im[:, : int(im.shape[1] * 0.40)]          # 60% of the width removed
        assert "heavily_cropped" in Q.assess(narrow).reasons


def test_a_uniform_field_is_blank_whatever_its_brightness():
    for level in (0.0, 0.5, 1.0):
        r = Q.assess(np.full((512, 512), level, dtype=np.float32))
        assert r.reasons == ["blank"], "level %.1f gave %s" % (level, r.reasons)


# ------------------------------------------------------------------ acceptances
def test_clean_images_pass():
    """The other half of the criterion: at most 1% of development may be rejected, so a
    clean image must not trip any rule."""
    for im in IMAGES:
        r = Q.assess(im)
        assert r.passed, "clean image rejected for %s (stats %s)" % (r.reasons, r.stats)


def test_mild_degradation_still_passes():
    """Real radiographs vary. A slightly dark or slightly bright image is a radiograph,
    not a fault, and rejecting it wastes a radiologist's time."""
    for im in IMAGES[:6]:
        for kind in ("under_exposed", "over_exposed"):
            assert Q.assess(Q.degrade(im, kind, severity=0.25)).passed, \
                "%s at severity 0.25 was rejected" % kind


def test_uint8_and_float_inputs_agree():
    im = IMAGES[0]
    a = Q.assess(im)
    b = Q.assess((im * 255).astype(np.uint8))
    assert a.passed == b.passed
    assert abs(a.stats["mean"] - b.stats["mean"]) < 2e-3


def test_three_channel_input_is_accepted():
    im = IMAGES[1]
    assert Q.assess(np.stack([im] * 3, axis=-1)).passed == Q.assess(im).passed


# ------------------------------------------------------------------ the vocabulary
def test_only_contract_reasons_are_ever_emitted():
    rng = np.random.default_rng(1)
    cases = [Q.degrade(im, k, s, rng) for im in IMAGES[:4]
             for k in ("clean", "black", "white", "under_exposed", "over_exposed",
                       "heavily_cropped")
             for s in (0.3, 0.7, 1.0)]
    cases += [np.zeros((512, 512)), np.ones((100, 900)), rng.random((300, 300))]
    for c in cases:
        for reason in Q.assess(c).reasons:
            assert reason in Q.REASONS, "%r is not one of the four C09 reasons" % reason


def test_assess_is_deterministic():
    im = IMAGES[3]
    assert Q.assess(im).reasons == Q.assess(im).reasons
    assert Q.assess(im).stats == Q.assess(im).stats


# ------------------------------------------------------------------ the thresholds
def test_thresholds_sit_in_a_real_gap_not_a_guess():
    """Every exposure threshold must separate clean images from degraded ones with
    nothing in between. If a distribution overlaps, no threshold satisfies both halves
    of the acceptance criteria and the rule needs rethinking, not tuning."""
    cal = Q.calibrate_thresholds(IMAGES)
    checks = [
        ("mean", "under_exposed", Q.DEFAULTS["dark_mean"]),
        ("mean", "over_exposed", Q.DEFAULTS["bright_mean"]),
        ("std", "black", Q.DEFAULTS["blank_std"]),
        ("std", "white", Q.DEFAULTS["blank_std"]),
    ]
    for stat, kind, threshold in checks:
        mid, separated = Q.gap_midpoint(cal["clean"][stat], cal[kind][stat],
                                        log=(stat == "std"))
        assert separated, "clean and %s overlap on %s; the rule cannot work" % (kind, stat)
        lo = min(min(cal["clean"][stat]), min(cal[kind][stat]))
        hi = max(max(cal["clean"][stat]), max(cal[kind][stat]))
        assert lo <= threshold <= hi, \
            "%s threshold %.3f is outside the observed range for %s" % (stat, threshold, kind)


def test_severity_is_monotone():
    """A worse image may not become acceptable again."""
    im = IMAGES[0]
    for kind in ("black", "white", "under_exposed", "over_exposed", "heavily_cropped"):
        passed = [Q.assess(Q.degrade(im, kind, s)).passed
                  for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert passed == sorted(passed, reverse=True), \
            "%s is not monotone in severity: %s" % (kind, passed)
