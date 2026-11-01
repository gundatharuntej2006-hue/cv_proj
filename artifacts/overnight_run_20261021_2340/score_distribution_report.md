# Score distributions by scanner: ranking, calibration and the referral threshold

**Read-only. No model was fitted and no threshold was chosen using held-out labels.**

Data: `error_analysis.csv`, `score_distribution_metrics.json`, `baseline_reproduction.csv`
Source file: `artifacts/p2/audit/vietnam_case_level_gate_audit.csv` — the 12×12-max
focal-pooling score for all 2,708 Vietnam participants, provenance verified from
`scripts/p2_35_gate_operational_audit.py` lines 72–101, not inferred from the file name.

**Threshold policy.** The single operating point used throughout is **2.4766**, the 70th
percentile of non-TB scores on the Vietnam **TUNE** half. TUNE is the declared development
partition. No threshold anywhere in this document was selected using REPORT labels, and
none was selected using Shenzhen or Montgomery labels.

---

## 1. The headline: the score scale shifts bodily between machines

Percentiles of the focal-pooling score, by scanner (5th / 25th / 50th / 75th / 95th):

| scanner | all cases | TB cases | **non-TB cases** |
|---|---|---|---|
| rad_system=1 | −4.10 / −2.06 / 0.67 / 2.76 / 4.83 | −2.48 / 1.33 / 2.86 / 3.52 / 5.38 | −4.14 / −2.19 / **0.27** / 2.40 / 4.64 |
| rad_system=2 | −4.01 / −2.57 / −0.18 / 2.44 / 4.86 | −1.21 / 1.11 / 2.82 / 4.23 / 6.12 | −4.05 / −2.78 / **−0.74** / 2.03 / 4.56 |
| rad_system=3 | −3.04 / 0.35 / 2.52 / 4.22 / 5.87 | 0.51 / 2.31 / 4.15 / 5.25 / 6.49 | −3.20 / 0.16 / **2.33** / 4.09 / 5.74 |

**The median healthy person on rad_system 3 scores 2.33. The shared referral threshold is
2.4766.** The median healthy person on rad_system 2 scores −0.74. That is a gap of **3.07
logits between two groups of healthy people**, larger than the interquartile range within
either group.

The TB distributions shift in the same direction by a similar amount (2.82 → 4.15), which
is why ranking survives while the operating point does not.

## 2. What that does at a single shared threshold

| scanner | n | TB | prevalence | **referred** | sensitivity | specificity | precision |
|---|---|---|---|---|---|---|---|
| rad_system=1 | 408 | 44 | 0.108 | 27.9% | 0.568 | 0.756 | 0.219 |
| rad_system=2 | 1,332 | 151 | 0.113 | 24.6% | 0.570 | **0.795** | 0.262 |
| rad_system=3 | 968 | 85 | **0.088** | **50.3%** | 0.741 | **0.520** | 0.129 |
| all | 2,708 | 280 | 0.103 | 34.3% | 0.621 | 0.689 | 0.187 |

**A patient imaged on rad_system 3 is referred at roughly twice the rate of a patient
imaged on rad_system 2 — 50.3% against 24.6% — while having the *lowest* tuberculosis
prevalence of the three groups (0.088 against 0.113).**

Specificity is 0.795 on one machine and 0.520 on another. In a screening programme that is
the difference between a workable confirmatory-testing load and an unworkable one, and it is
decided by which machine the patient walked past.

## 3. Separating ranking from calibration

These are different failures and the distinction matters for what to fix.

| scanner | AUROC (ranking) | spec at shared threshold | local threshold for 70% spec | sens at shared thr | sens at local thr |
|---|---|---|---|---|---|
| rad_system=1 | 0.7114 | 0.756 | 1.9832 | 0.568 | 0.636 |
| rad_system=2 | 0.7820 | 0.795 | 1.4852 | 0.570 | 0.709 |
| rad_system=3 | 0.7000 | 0.520 | 3.7262 | 0.741 | 0.588 |

**Ranking varies modestly: 0.700 to 0.782, a spread of 0.082.** The scanner-3 model is a
somewhat worse *discriminator*, but not catastrophically so.

**Calibration varies enormously.** The threshold that delivers 70% specificity is 1.49 on
rad_system 2 and 3.73 on rad_system 3 — a 2.24-logit difference for the same operating
intent.

The practical consequence: **most of the inequity in the table above is a threshold problem,
not a model problem.** Holding specificity fixed at 70% per machine, sensitivity becomes
0.636 / 0.709 / 0.588 — still unequal, but the referral-load blowout on rad_system 3
disappears entirely, with no change to the model.

## 4. Per-scanner estimates are noisier than they look

The same configuration, split across the two halves:

| scanner | TUNE AUROC | REPORT AUROC | swing |
|---|---|---|---|
| rad_system=1 | 0.7013 (n=204) | 0.7190 (n=204) | +0.018 |
| rad_system=2 | 0.7762 (n=665) | 0.7890 (n=667) | +0.013 |
| rad_system=3 | **0.7290** (n=483) | **0.6745** (n=485) | **−0.055** |
| pooled | 0.7237 | 0.7242 | +0.001 |

The pooled estimate is extremely stable across halves — selection optimism −0.0005 — while
the rad_system 3 estimate swings by 0.055 between two random halves of the same machine's
patients. **Any per-scanner comparison smaller than roughly 0.05 AUROC is inside the noise
of this cohort**, and a future source-robustness result must clear that bar on the worst
scanner, not on the pooled number.

## 5. The commercial comparator shows the same ordering

Lunit INSIGHT 3.9.0.1, on identical patients:

| scanner | our focal-pooling head | Lunit 3.9.0.1 | gap |
|---|---|---|---|
| rad_system=1 | 0.7114 | 0.7809 | 0.070 |
| rad_system=2 | 0.7820 | 0.8095 | 0.028 |
| rad_system=3 | 0.7000 | 0.7422 | 0.042 |
| all | 0.7240 | 0.7827 | 0.059 |

rad_system 3 is the hardest machine for both systems, so part of the degradation is a
property of those images rather than of our model. The gap to the commercial product is
nonetheless widest on rad_system 1 and narrowest on rad_system 2, so the two systems are not
simply tracking image difficulty.

## 6. The OOD gate repeats the same disparity

Fraction of cases the kNN gate would reject, by scanner: **rad_system 1 — 20.1%**,
rad_system 2 — 6.9%, rad_system 3 — 11.4%.

This reproduces the figure in `docs/p2_gate_operational_audit.md` from the case-level file.
It is a different disparity from the referral one above — the gate penalises rad_system 1
most, the threshold penalises rad_system 3 most — and the two are not correlated. That is
further reason not to treat the gate as a proxy for image quality.

## 7. What follows, and what does not

**Supported by this analysis.**
- Score distributions differ by acquisition source by more than the within-group spread.
- At a single global threshold, referral rate varies about twofold between machines, in the
  opposite direction to prevalence.
- Most of that is a calibration offset. Per-source thresholding removes the referral-load
  disparity without touching the model.

**Not supported.**
- That the model *uses* scanner information to decide. Distribution shift at the output is
  consistent with that, but also with a genuine difference in image characteristics that is
  correlated with the finding. The probe in `source_probe_report.md` shows the information is
  available; neither analysis shows it is used.
- Any population-level claim. Three machines, one cohort, one country.
- Anything clinical. This is a retrospective analysis of saved scores.
