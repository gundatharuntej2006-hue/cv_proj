# Probability Calibration & Temperature Scaling (Section 8, Table 12.2)

**Author:** Person 4 (`@tb-p4`)  
**Artifacts:** `artifacts/p4/calibration/`, `artifacts/p4/tables/t12_2_calibration.csv`

---

## 1. Motivation and Protocol
Deep neural networks and Vision Transformer (ViT) backbones commonly exhibit overconfident probability outputs under cross-entropy optimization. In safety-critical tuberculosis screening triage, calibrated probabilities are required so that downstream conformal risk control thresholds hold empirical coverage.

Per pre-registered governance rule **H6** and Decision **D22**, Platt Temperature Scaling is fitted **exclusively on the development split** after freezing classifier heads (D08–D11) and the kNN OOD gate (D13). The calibration split is never used for temperature fitting.

## 2. Formulation
For raw logit $z \in \mathbb{R}$, calibrated probability is computed via:
$$p_{\text{cal}} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + \exp(-z / T)}$$

Optimal temperature $T^*$ is obtained by minimizing negative log-likelihood (NLL) on the development cohort:
$$T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{dev}}} \sum_{i=1}^{N_{\text{dev}}} \left[ y_i \log \sigma\left(\frac{z_i}{T}\right) + (1 - y_i) \log \left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right]$$

## 3. Empirical Results
On the TBX11K internal test split ($N_{\text{tb}}=100, N_{\text{non\_tb}}=800$), temperature scaling with $T = 1.37$ substantially reduced Expected Calibration Error (ECE) across 10 equal-width bins:

| Metric | Point Estimate | 95% Bootstrap CI | Method |
|---|---|---|---|
| Uncalibrated ECE ($T=1.00$) | 0.0890 | [0.0620, 0.1180] | Bootstrap Percentile |
| Calibrated ECE ($T=1.37$) | 0.0340 | [0.0180, 0.0560] | Bootstrap Percentile |
