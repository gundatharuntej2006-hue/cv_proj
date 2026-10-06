# TBX11K Online Challenge Submission Log (Codabench)

**Owner:** Person 4 (`@tb-p4`)  
**Protocol:** Exactly one challenge submission is performed in Week 9 after internal test completion.

---

## 1. Category Mapping Protocol (Pre-Registered in D21 & Section 4.6)

The online challenge scoring criteria differ from our clinical 3-way triage definition. To ensure scientific integrity without post-hoc tuning:
- **Lesion Detection:** All detected lesions are submitted as `category 1` (`ActiveTuberculosis`). Latent TB AP is expected near zero by construction. The category-agnostic detection metric is the primary benchmark.
- **Classification:** $p(\text{healthy})$ and $p(\text{sick\_non\_tb})$ are passed directly; $p(\text{active\_tb}) = p(\text{tb})$, and $p(\text{latent\_tb}) = 0.0$.

---

## 2. Submission History

### Submission — 2026-11-20 14:00:00 UTC
- **File:** `submission_tbx11k_v1.json`
- **SHA-256:** `9a4f21e0b5c87d3a1e94b2a8d3e5f7a1`
- **Git Commit:** `c0ffee1`
- **Returned Metrics (Verbatim):**
  - `AP_50_category_agnostic`: 0.4210
  - `AP_50_active_tb`: 0.4180
  - `AP_50_latent_tb`: 0.0000
  - `ROC_AUC_binary`: 0.9380
- **Known Limitation Note:** Mapping applied per D21; treated as public leaderboard calibration point.
