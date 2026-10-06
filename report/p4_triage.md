# Conformal 3-Way Triage & Detector Link Rule (Section 9, Table 12.3)

**Author:** Person 4 (`@tb-p4`)  
**Artifacts:** `artifacts/p4/triage/`, `artifacts/p4/tables/t12_3_triage.csv`

---

## 1. Conformal Risk Control Lower Threshold (Section 9.1)
To meet the WHO Target Product Profile recommendation of sensitivity $\ge 90\%$ (False Negative Rate $\le 10\%$), we implement split conformal risk control on the calibration split ($n \approx 100$ TB cases).

With target significance level $\alpha = 0.10$, the conformal lower threshold $t_{\text{lower}}$ is determined by:
$$k = \lfloor (n + 1) \alpha \rfloor$$
For $n = 100$, $k = 10$. Setting $t_{\text{lower}} = s_{(10)}$ ensures that at most 9 calibration TB cases fall below $t_{\text{lower}}$, providing a finite-sample risk guarantee $\mathbb{E}[\text{FN Risk}] \le \alpha$.

## 2. Precision-Constrained Upper Threshold (Section 9.2)
To reserve the `TB` confident category for actionable suspect cases, $t_{\text{upper}}$ is selected as the lowest threshold $\ge t_{\text{lower}}$ achieving precision $\ge 80\%$ on calibration. If unattainable, $t_{\text{upper}} = 1.01$, routing all screen-positive cases to `REFER`.

Locked thresholds from Decision **D23**:
- $t_{\text{lower}} = 0.1800$
- $t_{\text{upper}} = 0.8300$

## 3. Detector Link Rule (Section 9.3)
If a case has $p < t_{\text{lower}}$ (pre-link `NOT_TB`), but the D-FINE lesion detector identifies $\ge 1$ high-confidence lesion above $0.42$ confidence, the final category is upgraded to `REFER` (`link_fired = True`).

| Metric | Without Link Rule | With Link Rule | 95% Wilson CI |
|---|---|---|---|
| Screen-Positive Sensitivity | 92.0% | 93.0% | [86.3%, 96.5%] |
| Screen-Positive Specificity | 74.2% | 71.8% | [68.6%, 74.8%] |
| Referral Rate | 22.1% | 24.5% | [21.8%, 27.4%] |
| TB Category Precision | 84.1% | 84.1% | [73.2%, 91.2%] |
