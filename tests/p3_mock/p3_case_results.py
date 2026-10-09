"""C14 CaseResult examples for P3 tests (P4's P4-02 fixtures are not in the repo yet).

Built from the README C14 example (ANALYSED) plus the null rules for REJECTED_GATE and UNREADABLE.
Replace with P4's fixtures when they land.
"""
from __future__ import annotations

import copy

DISCLAIMER = ("Decision support for TB triage only, not a diagnosis. Validated only on public research "
              "datasets; not prospectively clinically validated.")

ANALYSED = {
    "schema_version": "1.0", "case_id": "c_7f3a9e21", "status": "ANALYSED",
    "input": {"format": "dicom", "phi_tags_removed": 14, "error": None},
    "gate": {"quality_pass": True, "quality_reasons": [], "ood_score": 0.183, "ood_threshold": 0.241,
             "gate_pass": True, "view": None},
    "segmentation": {"sanity_pass": True, "fail_reasons": [], "crop_box": [10, 0, 507, 493],
                     "fallback_full": False},
    "classifier": {"model_id": "radino_crop_m0.10_clahe0_bin", "head": "binary", "p_tb_raw": 0.71,
                   "p_tb_cal": 0.64, "temperature": 1.37,
                   "three_class": {"healthy": 0.12, "sick_non_tb": 0.27, "tb": 0.61}},
    "detector": {"model_id": "dfine_512", "conf_threshold": 0.42, "n_lesions": 1,
                 "boxes": [{"xyxy": [301, 88, 372, 160], "score": 0.77}]},
    "triage": {"t_lower": 0.18, "t_upper": 0.83, "category_pre_link": "REFER", "link_fired": False,
               "category_final": "REFER", "screen_positive": True, "operating_point": "locked_default"},
    "explainability": {"heatmap_path": "runs/r0012/c_7f3a9e21_cam.npy",
                       "overlay_path": "runs/r0012/c_7f3a9e21_overlay.png"},
    "versions": {"model_version": "reg-v1-3b9c", "split_version": "v1",
                 "decisions_hash": "sha256:5e1d", "git_sha": "a1b2c3d"},
    "timing_ms": {"standardise": 40, "gate": 210, "segment": 95, "classify": 180, "detect": 260,
                  "explain": 420, "total": 1205},
    "disclaimer": DISCLAIMER,
}


def _variant(case_id: str, status: str, **blocks) -> dict:
    c = copy.deepcopy(ANALYSED)
    c.update(case_id=case_id, status=status)
    c["explainability"] = {"heatmap_path": f"runs/r0012/{case_id}_cam.npy",
                           "overlay_path": f"runs/r0012/{case_id}_overlay.png"}
    c.update(**blocks)
    return c


LINK_FIRED = _variant("c_0b11a7e3", "ANALYSED", triage={
    "t_lower": 0.18, "t_upper": 0.83, "category_pre_link": "NOT_TB", "link_fired": True,
    "category_final": "REFER", "screen_positive": True, "operating_point": "locked_default"})
REJECTED = _variant("c_9a2c44d0", "REJECTED_GATE", segmentation=None, classifier=None, detector=None,
                    triage=None, gate={**ANALYSED["gate"], "gate_pass": False, "ood_score": 0.51})
UNREADABLE = _variant("c_31f0e9b5", "UNREADABLE", gate=None, segmentation=None, classifier=None,
                      detector=None, triage=None, explainability=None, timing_ms=None,
                      input={"format": "png", "phi_tags_removed": 0, "error": "cannot decode image"})

CASES = {"analysed": ANALYSED, "link_fired": LINK_FIRED, "rejected_gate": REJECTED, "unreadable": UNREADABLE}


# The 7 scenario types of P4-02 (TB, REFER, NOT TB, link fired, gate rejected, unreadable,
# segmentation fallback): stand-ins until P4's real fixtures exist.
TB_CASE = _variant("c_51d0aa42", "ANALYSED",
                   classifier={**ANALYSED["classifier"], "p_tb_raw": 0.93, "p_tb_cal": 0.91},
                   triage={**ANALYSED["triage"], "category_pre_link": "TB", "category_final": "TB"})
NOT_TB_CASE = _variant("c_e2b7c019", "ANALYSED",
                       classifier={**ANALYSED["classifier"], "p_tb_raw": 0.06, "p_tb_cal": 0.04},
                       detector={**ANALYSED["detector"], "n_lesions": 0, "boxes": []},
                       triage={**ANALYSED["triage"], "category_pre_link": "NOT_TB", "category_final": "NOT_TB",
                               "screen_positive": False})
SEG_FALLBACK = _variant("c_7c3e5d18", "ANALYSED",
                        segmentation={"sanity_pass": False, "fail_reasons": ["area"], "crop_box": [0, 0, 512, 512],
                                      "fallback_full": True})

P4_SCENARIOS = {"tb": TB_CASE, "refer": ANALYSED, "not_tb": NOT_TB_CASE, "link_fired": LINK_FIRED,
                "rejected_gate": REJECTED, "unreadable": UNREADABLE, "segmentation_fallback": SEG_FALLBACK}
