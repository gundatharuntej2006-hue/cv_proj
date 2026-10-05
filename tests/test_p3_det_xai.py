from pathlib import Path
import numpy as np
from p3_det.dfine_detector import DFINEDetector
from p3_xai.gradcam import GradCAMExplainer
from p3_ops.pdf_generator import TriagePDFReportGenerator
from p2_cls.classifier import TBClassifier
from tbcore.schemas import PipelineResult


def test_p3_detector():
    det = DFINEDetector()
    img = np.zeros((512, 512), dtype=np.float32)
    res = det.detect(img, "IMG_P3")
    assert res.total_lesions_found >= 0


def test_p3_xai():
    xai = GradCAMExplainer()
    clf = TBClassifier()
    img = np.zeros((512, 512), dtype=np.float32)
    res = xai.generate(clf, img, "IMG_P3")
    assert len(res.max_activation_coords) == 2


def test_p3_pdf(tmp_path):
    gen = TriagePDFReportGenerator()
    with open("mock/sample_pipeline_packet.json") as f:
        import json
        packet = PipelineResult(**json.load(f))
    out = tmp_path / "report.pdf"
    res_path = gen.generate_report(packet, str(out))
    assert Path(res_path).exists()
