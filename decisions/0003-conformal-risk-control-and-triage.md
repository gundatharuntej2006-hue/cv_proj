# ADR 0003: Conformal Risk Control for Clinical Triage

## Status
Accepted

## Context
Standard binary classification (thresholding at 0.5) offers no mathematical guarantee on patient safety. In resource-limited TB screening camps, missing a active TB case (false negative) can lead to severe mortality and transmission. WHO TPP mandates sensitivity >= 90% and miss rate <= 10%.

## Decision
We adopt Conformal Risk Control (Angelopoulos et al., 2022) to formulate a three-way triage policy:
1. `TB`: High confidence active pulmonary TB; priority clinician review.
2. `REFER`: Indeterminate / borderline risk where prediction uncertainty is high; flagged for radiologist review or GeneXpert sputum testing.
3. `NOT_TB`: High confidence negative, screened out with bounded false negative rate:
$$P(Y=1 \mid \text{Triage} = \text{NOT_TB}) \le \alpha = 0.10$$

## Consequences
- Clinicians gain transparent uncertainty signals (`REFER` bucket).
- Triage sensitivity is mathematically guaranteed on exchangeable unseen data.
