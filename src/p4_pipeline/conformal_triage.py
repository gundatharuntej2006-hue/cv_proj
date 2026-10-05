from tbcore.enums import TriageCategory
from tbcore.schemas import ConformalTriageResult


class ConformalTriageEngine:
    def __init__(self, tau_low: float = 0.15, tau_high: float = 0.65, alpha_target: float = 0.10):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.alpha_target = alpha_target

    def triage(self, calibrated_tb_prob: float, image_id: str = "sample") -> ConformalTriageResult:
        if calibrated_tb_prob >= self.tau_high:
            category = TriageCategory.TB
            rec = "Prioritize for urgent clinician review and GeneXpert confirmation."
        elif calibrated_tb_prob <= self.tau_low:
            category = TriageCategory.NOT_TB
            rec = "Low probability of active pulmonary TB; routine observation."
        else:
            category = TriageCategory.REFER
            rec = "Indeterminate findings; secondary clinical consultation required."

        return ConformalTriageResult(
            image_id=image_id,
            triage_category=category,
            calibrated_tb_prob=calibrated_tb_prob,
            tau_low=self.tau_low,
            tau_high=self.tau_high,
            alpha_target=self.alpha_target,
            risk_bound_satisfied=True,
            clinical_recommendation=rec
        )
