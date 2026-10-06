"""Single source of clinical disclaimer and scope honesty text for cv_proj."""

DISCLAIMER_TEXT = (
    "Decision support for TB triage only, not a diagnosis. "
    "Validated only on public research datasets; not prospectively clinically validated."
)

SCOPE_HONESTY_TEXT = (
    "Scope & Limitations: This Computer-Aided Detection (CAD) system is designed strictly "
    "for pulmonary tuberculosis screening triage in resource-constrained community settings. "
    "It is intended to prioritize suspect radiographs for confirmatory microbiological or expert "
    "radiologist review, not to replace clinical diagnostic judgement. Performance has been "
    "characterized on retrospective benchmark cohorts and requires prospective clinical validation "
    "before deployment."
)

WHO_TARGET_INFO = (
    "Target Product Profile: WHO High-Priority CAD Triage Target — Sensitivity >= 90%, Specificity >= 70%."
)
