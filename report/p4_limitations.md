# Limitations, Scope Honesty & Governance Audit (Section 21)

**Author:** Person 4 (`@tb-p4`)  
**Scope:** Honest clinical limitations, known technical caveats, and no-test-tuning gate audit.

---

## 1. Scope & Clinical Boundaries
- **Decision Support Only:** This CAD system is engineered exclusively for community screening triage prioritization, not for standalone clinical diagnosis.
- **Retrospective Data Constraint:** The system was evaluated on retrospective open-access research benchmark datasets (TBX11K, Shenzhen, Montgomery). It has **not** been prospectively clinically validated in field deployment.
- **Microbiological Confirmation:** Radiographic triage suspicious of TB must be confirmed via sputum smear microscopy, GeneXpert MTB/RIF, or mycobacterial culture.

## 2. Technical Limitations
- **Latent TB Lesions:** Due to severe training dataset sparsity (only 212 latent TB cases in TBX11K), latent lesion detection performance is not clinically reliable.
- **External Calibration Transfer:** While the conformal risk control bounds guarantee $\le 10\%$ false negatives under exchangeability, domain shift across external clinical acquisition centers requires local threshold recalibration (Section 13 / P4-17).
- **Detector Contingency:** In the event of detector AP50 falling below 0.05 on development, the link rule is automatically disabled and reported as a negative finding, preserving classifier integrity.

## 3. Governance Audit Summary
- All 23 decision files (D01–D23) were locked prior to sealed data evaluation.
- No model weights or hyperparameters were adjusted after single-use token issuance.
- Challenge submission was conducted exactly once per Section 4.6 category mapping rules.
