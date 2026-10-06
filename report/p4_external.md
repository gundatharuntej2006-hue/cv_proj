# External Validation Protocol (Section 13, Table 13)

**Author:** Person 4 (`@tb-p4`)  
**Artifacts:** `artifacts/p4/tables/t13_external.csv`, `artifacts/p4/tables/t13_recal.csv`

---

## 1. External Cohort Overview
To assess domain generalizability and domain shift robustness, frozen models were evaluated without fine-tuning on two independent cohorts:
- **NLM Shenzhen Hospital:** $N = 662$ ($336$ TB cases, $326$ non-TB controls).
- **NLM Montgomery County:** $N = 138$ ($58$ TB cases, $80$ non-TB controls).

## 2. Results and Conformal Risk Coverage

| Cohort | Metric | Point Estimate | 95% CI | Sample Size |
|---|---|---|---|---|
| Shenzhen | ROC-AUC | 0.918 | [0.894, 0.940] | 336 TB, 326 Non-TB |
| Shenzhen | Conformal Sensitivity ($t_{\text{low}}=0.18$) | 89.6% | [85.9%, 92.4%] | 336 TB, 326 Non-TB |
| Shenzhen | Conformal Specificity | 71.2% | [66.0%, 75.8%] | 336 TB, 326 Non-TB |
| Montgomery | ROC-AUC | 0.884 | [0.824, 0.938] | 58 TB, 80 Non-TB |
| Montgomery | Conformal Sensitivity | 87.9% | [77.1%, 94.0%] | 58 TB, 80 Non-TB |
| Montgomery | Conformal Specificity | 68.8% | [58.0%, 77.8%] | 58 TB, 80 Non-TB |

*Montgomery Caveat:* The Montgomery cohort has only 58 TB cases, resulting in wider confidence intervals. Under exchangeability violation, conformal miss rate slightly exceeds $\alpha=0.10$, demonstrating the necessity of local calibration (P4-17 extension).
