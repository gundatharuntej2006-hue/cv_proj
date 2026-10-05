# WHO Target Product Profile (TPP) Compliance Guide

The World Health Organization (WHO) provides standardized evaluation criteria for Computer-Aided Detection (CAD) software for TB screening:

| Metric | WHO Screening TPP Minimum | WHO TPP Optimal | cv_proj Project Target |
| :--- | :--- | :--- | :--- |
| **Sensitivity** | >= 90% | >= 95% | **>= 90%** (validated with 95% bootstrap CI) |
| **Specificity** | >= 70% | >= 80% | **>= 70%** (validated with 95% bootstrap CI) |
| **Conformal Miss Rate** | <= 10% | <= 5% | **<= 10%** (at alpha = 0.10) |
| **Turnaround Latency** | <= 10 seconds | <= 2 seconds | **<= 2.0 seconds** on laptop GPU |
| **Explainability** | Visual explanation required | Lesion localization | **Grad-CAM + D-FINE boxes** |
