"""Mock generator for Person 4 contracts (C12, C13) and the 7 C14 CaseResult fixtures."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from tbcore.disclaimer import DISCLAIMER_TEXT
from tbcore.enums import CaseStatus
from tbcore.paths import get_repo_root, resolve_path
from tbcore.schemas import (
    CaseResult,
    ClassifierBlock,
    DetectedBox,
    DetectorBlock,
    ExplainabilityBlock,
    GateBlock,
    InputBlock,
    SegmentationBlock,
    ThreeClassProbs,
    TimingBlock,
    TriageBlock,
    VersionsBlock,
)


def generate_all_p4_mocks(base_dir: Optional[Path] = None):
    if base_dir is None:
        base_dir = get_repo_root()

    mock_dir = base_dir / "mock"
    p4_triage_dir = mock_dir / "artifacts" / "p4" / "triage"
    p4_triage_dir.mkdir(parents=True, exist_ok=True)

    fixtures_dir = mock_dir / "fixtures" / "case_results"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    cases_dir = mock_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. C12 Mock: triage_development_v1.parquet
    # -----------------------------------------------------------------------
    c12_rows = []
    for i in range(660):
        is_tb = (i < 60)
        p_cal = float(np.random.beta(5, 2) if is_tb else np.random.beta(2, 5))
        n_les = int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))

        if p_cal < 0.18:
            cat_pre = "NOT_TB"
            link = (n_les >= 1)
            cat_final = "REFER" if link else "NOT_TB"
        elif p_cal < 0.83:
            cat_pre = "REFER"
            link = False
            cat_final = "REFER"
        else:
            cat_pre = "TB"
            link = False
            cat_final = "TB"

        c12_rows.append({
            "image_id": f"dev_{i:04d}",
            "split": "development",
            "model_id": "radino_crop_m0.10_clahe0_bin",
            "p_tb_cal": p_cal,
            "temperature": 1.37,
            "t_lower": 0.18,
            "t_upper": 0.83,
            "category_pre_link": cat_pre,
            "link_fired": link,
            "category_final": cat_final,
            "screen_positive": cat_final in ("TB", "REFER"),
            "gate_pass": True,
            "sanity_pass": True,
            "n_lesions": n_les,
        })
    df_c12 = pd.DataFrame(c12_rows)
    df_c12.to_parquet(p4_triage_dir / "triage_development_v1.parquet", index=False)
    # Also save as CSV for quick viewing
    df_c12.to_csv(p4_triage_dir / "triage_development_v1.csv", index=False)

    # -----------------------------------------------------------------------
    # 2. C13 Mock: operating_curve_calibration_v1.parquet
    # -----------------------------------------------------------------------
    c13_rows = []
    grid = np.linspace(0.01, 0.99, 50)
    if 0.18 not in grid:
        grid = np.sort(np.append(grid, 0.18))

    for t in grid:
        sens = float(max(0.0, min(1.0, 1.0 - 0.5 * t)))
        spec = float(max(0.0, min(1.0, 0.4 + 0.58 * t)))
        c13_rows.append({
            "t_lower": float(t),
            "sensitivity": sens,
            "specificity": spec,
            "refer_rate": float(0.35 - 0.2 * t),
            "tb_rate": float(0.15 - 0.1 * t),
            "n_tb": 100,
            "n_non_tb": 800,
            "is_locked_default": bool(np.isclose(t, 0.18, atol=1e-4)),
        })
    df_c13 = pd.DataFrame(c13_rows)
    df_c13.to_parquet(p4_triage_dir / "operating_curve_calibration_v1.parquet", index=False)
    df_c13.to_csv(p4_triage_dir / "operating_curve_calibration_v1.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. 7 CaseResult Fixtures (C14)
    # -----------------------------------------------------------------------
    v = VersionsBlock(model_version="reg-v1-3b9c", split_version="v1", decisions_hash="sha256:5e1d4b2", git_sha="c0ffee1")
    t_std = TimingBlock(standardise=35, gate=180, segment=90, classify=160, detect=240, explain=380, total=1085)

    # Fixture 1: Confirmed TB Positive
    f1 = CaseResult(
        schema_version="1.0",
        case_id="c_tb_pos_001",
        status=CaseStatus.ANALYSED,
        input=InputBlock(format="dicom", phi_tags_removed=12, error=None),
        gate=GateBlock(quality_pass=True, quality_reasons=[], ood_score=0.142, ood_threshold=0.241, gate_pass=True),
        segmentation=SegmentationBlock(sanity_pass=True, fail_reasons=[], crop_box=[20, 15, 492, 480], fallback_full=False),
        classifier=ClassifierBlock(
            model_id="radino_crop_m0.10_clahe0_bin",
            head="binary",
            p_tb_raw=0.91,
            p_tb_cal=0.88,
            temperature=1.37,
            three_class=ThreeClassProbs(healthy=0.03, sick_non_tb=0.10, tb=0.87)
        ),
        detector=DetectorBlock(
            model_id="dfine_512",
            conf_threshold=0.42,
            n_lesions=2,
            boxes=[
                DetectedBox(xyxy=[120, 85, 210, 175], score=0.84),
                DetectedBox(xyxy=[310, 140, 390, 220], score=0.72),
            ]
        ),
        triage=TriageBlock(t_lower=0.18, t_upper=0.83, category_pre_link="TB", link_fired=False, category_final="TB", screen_positive=True, operating_point="locked_default"),
        explainability=ExplainabilityBlock(heatmap_path="runs/r001/c_tb_pos_001_cam.npy", overlay_path="runs/r001/c_tb_pos_001_overlay.png"),
        versions=v,
        timing_ms=t_std,
        disclaimer=DISCLAIMER_TEXT,
    )

    # Fixture 2: REFER (Indeterminate risk)
    f2 = CaseResult(
        schema_version="1.0",
        case_id="c_refer_002",
        status=CaseStatus.ANALYSED,
        input=InputBlock(format="png", phi_tags_removed=0, error=None),
        gate=GateBlock(quality_pass=True, quality_reasons=[], ood_score=0.195, ood_threshold=0.241, gate_pass=True),
        segmentation=SegmentationBlock(sanity_pass=True, fail_reasons=[], crop_box=[10, 0, 507, 493], fallback_full=False),
        classifier=ClassifierBlock(
            model_id="radino_crop_m0.10_clahe0_bin",
            head="binary",
            p_tb_raw=0.52,
            p_tb_cal=0.48,
            temperature=1.37,
            three_class=ThreeClassProbs(healthy=0.25, sick_non_tb=0.35, tb=0.40)
        ),
        detector=DetectorBlock(model_id="dfine_512", conf_threshold=0.42, n_lesions=0, boxes=[]),
        triage=TriageBlock(t_lower=0.18, t_upper=0.83, category_pre_link="REFER", link_fired=False, category_final="REFER", screen_positive=True, operating_point="locked_default"),
        explainability=ExplainabilityBlock(heatmap_path="runs/r001/c_refer_002_cam.npy", overlay_path="runs/r001/c_refer_002_overlay.png"),
        versions=v,
        timing_ms=t_std,
        disclaimer=DISCLAIMER_TEXT,
    )

    # Fixture 3: Clear NOT_TB Negative
    f3 = CaseResult(
        schema_version="1.0",
        case_id="c_not_tb_003",
        status=CaseStatus.ANALYSED,
        input=InputBlock(format="png", phi_tags_removed=0, error=None),
        gate=GateBlock(quality_pass=True, quality_reasons=[], ood_score=0.088, ood_threshold=0.241, gate_pass=True),
        segmentation=SegmentationBlock(sanity_pass=True, fail_reasons=[], crop_box=[15, 10, 500, 490], fallback_full=False),
        classifier=ClassifierBlock(
            model_id="radino_crop_m0.10_clahe0_bin",
            head="binary",
            p_tb_raw=0.08,
            p_tb_cal=0.06,
            temperature=1.37,
            three_class=ThreeClassProbs(healthy=0.92, sick_non_tb=0.05, tb=0.03)
        ),
        detector=DetectorBlock(model_id="dfine_512", conf_threshold=0.42, n_lesions=0, boxes=[]),
        triage=TriageBlock(t_lower=0.18, t_upper=0.83, category_pre_link="NOT_TB", link_fired=False, category_final="NOT_TB", screen_positive=False, operating_point="locked_default"),
        explainability=ExplainabilityBlock(heatmap_path="runs/r001/c_not_tb_003_cam.npy", overlay_path="runs/r001/c_not_tb_003_overlay.png"),
        versions=v,
        timing_ms=t_std,
        disclaimer=DISCLAIMER_TEXT,
    )

    # Fixture 4: Detector Link Rule Fired (NOT_TB -> REFER)
    f4 = CaseResult(
        schema_version="1.0",
        case_id="c_link_fired_004",
        status=CaseStatus.ANALYSED,
        input=InputBlock(format="png", phi_tags_removed=0, error=None),
        gate=GateBlock(quality_pass=True, quality_reasons=[], ood_score=0.155, ood_threshold=0.241, gate_pass=True),
        segmentation=SegmentationBlock(sanity_pass=True, fail_reasons=[], crop_box=[10, 5, 505, 495], fallback_full=False),
        classifier=ClassifierBlock(
            model_id="radino_crop_m0.10_clahe0_bin",
            head="binary",
            p_tb_raw=0.14,
            p_tb_cal=0.12,
            temperature=1.37,
            three_class=ThreeClassProbs(healthy=0.78, sick_non_tb=0.15, tb=0.07)
        ),
        detector=DetectorBlock(
            model_id="dfine_512",
            conf_threshold=0.42,
            n_lesions=1,
            boxes=[DetectedBox(xyxy=[340, 90, 410, 160], score=0.79)]
        ),
        triage=TriageBlock(t_lower=0.18, t_upper=0.83, category_pre_link="NOT_TB", link_fired=True, category_final="REFER", screen_positive=True, operating_point="locked_default"),
        explainability=ExplainabilityBlock(heatmap_path="runs/r001/c_link_fired_004_cam.npy", overlay_path="runs/r001/c_link_fired_004_overlay.png"),
        versions=v,
        timing_ms=t_std,
        disclaimer=DISCLAIMER_TEXT,
    )

    # Fixture 5: Quality Gate Rejected (Non-medical / Lateral / Blurred)
    f5 = CaseResult(
        schema_version="1.0",
        case_id="c_gate_rejected_005",
        status=CaseStatus.REJECTED_GATE,
        input=InputBlock(format="png", phi_tags_removed=0, error=None),
        gate=GateBlock(
            quality_pass=False,
            quality_reasons=["over_exposed", "heavily_cropped"],
            ood_score=0.412,
            ood_threshold=0.241,
            gate_pass=False,
            view="lateral_or_other"
        ),
        segmentation=None,
        classifier=None,
        detector=None,
        triage=None,
        explainability=None,
        versions=v,
        timing_ms=TimingBlock(standardise=30, gate=190, total=220),
        disclaimer=DISCLAIMER_TEXT,
    )

    # Fixture 6: Corrupt / Unreadable Image
    f6 = CaseResult(
        schema_version="1.0",
        case_id="c_unreadable_006",
        status=CaseStatus.UNREADABLE,
        input=InputBlock(format="dicom", phi_tags_removed=0, error="Invalid DICOM header: EOF while reading PixelData"),
        gate=None,
        segmentation=None,
        classifier=None,
        detector=None,
        triage=None,
        explainability=None,
        versions=v,
        timing_ms=TimingBlock(standardise=15, total=15),
        disclaimer=DISCLAIMER_TEXT,
    )

    # Fixture 7: Segmentation Sanity Fallback to Full Image
    f7 = CaseResult(
        schema_version="1.0",
        case_id="c_fallback_full_007",
        status=CaseStatus.ANALYSED,
        input=InputBlock(format="png", phi_tags_removed=0, error=None),
        gate=GateBlock(quality_pass=True, quality_reasons=[], ood_score=0.170, ood_threshold=0.241, gate_pass=True),
        segmentation=SegmentationBlock(
            sanity_pass=False,
            fail_reasons=["lung_area_too_small", "single_lung_component"],
            crop_box=[0, 0, 512, 512],
            fallback_full=True
        ),
        classifier=ClassifierBlock(
            model_id="radino_full_clahe0_bin",
            head="binary",
            p_tb_raw=0.68,
            p_tb_cal=0.62,
            temperature=1.37,
            three_class=ThreeClassProbs(healthy=0.15, sick_non_tb=0.25, tb=0.60)
        ),
        detector=DetectorBlock(model_id="dfine_512", conf_threshold=0.42, n_lesions=0, boxes=[]),
        triage=TriageBlock(t_lower=0.18, t_upper=0.83, category_pre_link="REFER", link_fired=False, category_final="REFER", screen_positive=True, operating_point="locked_default"),
        explainability=ExplainabilityBlock(heatmap_path="runs/r001/c_fallback_full_007_cam.npy", overlay_path="runs/r001/c_fallback_full_007_overlay.png"),
        versions=v,
        timing_ms=t_std,
        disclaimer=DISCLAIMER_TEXT,
    )

    fixtures = [
        ("01_tb_positive.json", f1),
        ("02_refer.json", f2),
        ("03_not_tb.json", f3),
        ("04_link_fired.json", f4),
        ("05_rejected_gate.json", f5),
        ("06_unreadable.json", f6),
        ("07_fallback_full.json", f7),
    ]

    for fname, obj in fixtures:
        raw_dict = obj.model_dump(mode="json")
        for target_folder in [fixtures_dir, cases_dir]:
            with open(target_folder / fname, "w", encoding="utf-8") as fp:
                json.dump(raw_dict, fp, indent=2)


if __name__ == "__main__":
    generate_all_p4_mocks()
    print("Generated all P4 mock files and CaseResult fixtures.")
