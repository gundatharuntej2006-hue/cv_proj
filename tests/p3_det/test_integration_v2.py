from types import SimpleNamespace

import numpy as np
import pytest

from p3_det.integration_v2 import analyse, enrich, referral
from p3_det.result import DetResult


class Detector:
    def detect(self, image, crop):
        assert image.shape == (512, 512)
        return DetResult(np.array([[10, 20, 50, 60], [80, 90, 110, 120]], np.float32),
                         np.array([0.9, 0.2], np.float32), 1)

    def detector_info(self):
        return {"model_id": "test", "conf_threshold": 0.5, "mock": False, "locked": True}


def test_real_box_score_arrays_and_threshold():
    out = analyse(np.zeros((512, 512), np.uint8), detector=Detector())
    assert out["n_lesions"] == 1
    assert out["boxes"][0]["xyxy"] == [10, 20, 50, 60]


@pytest.mark.parametrize("category,expected", [("NOT_TB", "REFER"), ("REFER", "REFER"), ("TB", "TB")])
def test_link_rule_only_upgrades(category, expected):
    assert referral(category, 1) == expected
    assert referral(category, 0) == category


def test_v2_scores_and_flags_preserved_and_failures_distinguished():
    case = {"status": "ANALYSED", "analysis": {"image_logit": -0.5,
            "operating_points": {"screen_positive_at_spec_0.70": False}}}
    image = np.zeros((512, 512), np.uint8)
    out = enrich(case, image, detector=Detector())
    assert out["analysis"]["detector_referral"]["additional_review"]
    assert out["analysis"]["image_logit"] == -0.5
    assert "detector" not in case["analysis"]
    broken = SimpleNamespace(detect=lambda *a: (_ for _ in ()).throw(RuntimeError("missing checkpoint")))
    out = enrich(case, image, detector=broken)
    assert out["analysis"]["detector"]["n_lesions"] is None
    assert not out["analysis"]["detector_referral"]["additional_review"]


def test_rejected_cases_stay_unscored():
    case = {"status": "REJECTED", "analysis": None}
    assert enrich(case, None) == case


def test_mock_cannot_be_attached_to_real_v2():
    detector = Detector()
    detector.detector_info = lambda: {"mock": True}
    with pytest.raises(RuntimeError, match="Mock"):
        analyse(np.zeros((512, 512), np.uint8), detector=detector)


def test_service_hook_covers_analysis_overlay_and_ranked_batch(monkeypatch):
    import io
    from PIL import Image
    from p3_det import integration_v2 as bridge
    monkeypatch.setattr(bridge.api, "detect", Detector().detect)
    monkeypatch.setattr(bridge.api, "detector_info", Detector().detector_info)
    case = {"status": "ANALYSED", "image": "private_name.png", "latency_ms": 1,
            "analysis": {"image_logit": -0.5,
                         "operating_points": {"screen_positive_at_spec_0.70": False}}}
    buf = io.BytesIO()
    Image.new("L", (512, 512)).save(buf, format="PNG")
    raw = buf.getvalue()
    service = SimpleNamespace(analyse_bytes_v2=lambda *a, **kw: case,
                              overlay_png=lambda *a: raw,
                              rank_batch=lambda results: {"cases": [{"rank": 1}]},
                              p3_input_provider=lambda image: (image, [0, 0, 512, 512]))
    bridge.install(service)
    out = service.analyse_bytes_v2(raw)
    assert out["analysis"]["detector"]["n_lesions"] == 1
    ranked = service.rank_batch([out])
    assert ranked["n_additional_detector_review"] == 1
    assert ranked["cases"][0]["detector"]["model_id"] == "test"
    overlay = Image.open(io.BytesIO(service.overlay_png(raw, out["analysis"])))
    assert overlay.getpixel((10, 20)) == (0, 220, 255)
    assert "detector" not in case["analysis"]
