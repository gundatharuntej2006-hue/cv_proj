# Evaluation Protocol, Statistical Methodology & Challenge Benchmark (Section 12)

**Author:** Person 4 (`@tb-p4`)  
**Artifacts:** `artifacts/p4/tables/`, `decisions/p4_analysis_plan.json`

---

## 1. Statistical Confidence Intervals & Sample Sizes
In compliance with Section 12 governance rules, every reported metric is accompanied by:
1. Exact sample size breakdown: positive cases ($n_{\text{tb}}$) and negative cases ($n_{\text{non\_tb}}$).
2. Two-sided 95% Confidence Intervals:
   - **Wilson Score Interval** for binomial proportions (Sensitivity, Specificity, PPV, NPV, Referral Rate).
   - **Stratified Bootstrap Percentile Interval** ($B = 2,000$ replicates, fixed random seed 42) for continuous summary statistics (ROC-AUC, ECE, AP@0.50).

## 2. Table Summary (Internal Test Split: $N=900$)

| Table ID | Description | Key Metric | 95% CI |
|---|---|---|---|
| `t12_1` | Primary Classification | ROC-AUC: 0.942 | [0.915, 0.968] |
| `t12_1` | WHO Target Sensitivity @ 70% Spec | Sens: 92.0% | [85.0%, 95.8%] |
| `t12_2` | Expected Calibration Error | ECE: 0.034 | [0.018, 0.056] |
| `t12_3` | Conformal Screen-Positive Sens | Sens: 93.0% | [86.3%, 96.5%] |
| `t12_6` | OOD Gate Rejection on Lateral/Other | Rej: 96.2% | [94.2%, 97.6%] |
| `t12_7` | Lung Segmentation Sanity Failure | Fail: 1.67% | [1.02%, 2.72%] |
| `t12_8` | Challenge Submission Leaderboard | Cat-Agnostic AP50: 0.421 | Leaderboard verified |
