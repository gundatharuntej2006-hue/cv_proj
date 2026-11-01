# Prior-art status and candidate contributions

**No new literature was retrieved during this run.** The runbook prefers offline work, and
every source below was already retrieved and recorded in
`artifacts/p2/audit/SOURCES.md` during an earlier session. This document does not restate
their content; it records their verification status and asks what, after tonight's
measurements, is actually defensible as a contribution.

## Verification status, unchanged

| source | status | usable for a claim? |
|---|---|---|
| **RADAR** — *Acquisition-Adversarial Attention Pooling over a Frozen Chest-Radiograph Foundation Model for Tuberculosis Screening*, MICCAI 2026 satellite | retrieved and read | yes |
| **RAD-DINO model card**, Hugging Face | retrieved and read | yes |
| **CheXFound / GLoRI**, IEEE TMI 44(12) 2025 | **not retrieved** | no — context only, no specific claim |
| *Domain-Aware TB screening with a LoRA-adapted CXR foundation model*, MICCAI 2026 satellite | page retrieved, bibliographic details unconfirmed | **excluded from the paper draft** |
| *Shortcut learning in medical AI hinders generalization*, npj Digital Medicine 2024 | **not retrieved** | no — the project's shortcut findings rest on its own measurements |

**The architecture is not novel and no document in this project claims it is.** RADAR covers
patch-level aggregation over a frozen chest-radiograph foundation model for tuberculosis,
and additionally applies adversarial and style-randomisation *mitigation* that this project
does not have.

## Candidate contributions, after tonight

The runbook asks for at most one or two supported by results. Two survive.

### 1. A measured acquisition-source audit of a frozen CXR foundation model on a consecutive screening cohort

Three measurements, taken tonight, that together form a single coherent finding:

- source is **97.4% balanced-accuracy decodable** from the frozen representation
  (chance 33.8%) on a cohort the backbone never saw;
- source is nonetheless **nearly independent of the outcome** in that cohort
  (χ² = 4.051, p = 0.132, Cramér's V = 0.039);
- at one shared referral threshold, **referral rate varies roughly twofold between
  machines** — 50.3% against 24.6% — in the *opposite* direction to prevalence, and
  specificity falls from 0.795 to 0.520.

RADAR mitigates acquisition shortcuts; it does not, on the page retrieved, report a
quantified per-scanner operating-point audit on a consecutive Xpert-confirmed cohort with a
commercial comparator on identical patients. **The audit, not the architecture, is the
candidate contribution.** It is also the kind of result that stays useful when the model is
replaced.

Honest limitation: three machines, one cohort, one country. This is a worked case study, not
a general claim about acquisition hardware.

### 2. The decomposition of that disparity into ranking versus calibration

Per-scanner AUROC spans 0.700–0.782, a spread of 0.082. Per-scanner thresholds for the same
70% specificity span 1.49–3.73 logits. **Holding specificity fixed per machine removes the
referral-load disparity entirely with no change to the model.**

This matters because the field's usual response to a source disparity is a representation
intervention. Here the larger share is an operating-point problem with a cheaper fix, and
the project can say so with numbers.

Honest limitation: per-scanner AUROC swings by up to 0.055 between two random halves of the
same machine's patients (rad_system 3: TUNE 0.7290, REPORT 0.6745). The ranking spread is
therefore only modestly larger than the noise, while the calibration gap is far outside it.
The decomposition is well supported in direction; the exact ranking numbers are not precise.

## What is not a contribution

- The 12×12 regional max-pooling aggregator itself — prior art, see above.
- The Vietnam, Shenzhen and Montgomery numbers as independent generalisation estimates —
  all three cohorts influenced earlier configuration choices.
- Anything described as clinically validated. Nothing here is prospective.
