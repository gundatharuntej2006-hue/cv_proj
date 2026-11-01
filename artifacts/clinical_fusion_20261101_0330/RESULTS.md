# Clinical-context fusion — results

**Run:** `artifacts/clinical_fusion_20261101_0330/` · 2026-11-01 · 4.3 min
**Protocol sha-256:** `2ee3e4b8…`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged; `git status` identical
**Exploratory.** Vietnam TUNE is exposed — it selected the pooling configuration — so the
absolute AUROCs are optimistic. The fusion *increment* was fitted and scored entirely
inside 10-fold × 20-repeat cross-validation and is the honest quantity. **Nothing is
adopted into production by this run.**

---

## Headline

> **Fusing the frozen image score with the clinical fields a screening form already
> collects raises AUROC from 0.7237 to 0.7809 — Δ +0.058 [+0.030, +0.085] — and in the
> previous-TB population from 0.6144 to 0.7122 — Δ +0.100 [+0.035, +0.165].**
>
> **The same fusion does nothing for the commercial comparator** (Lunit 3.9.0.1:
> 0.7780 → 0.7778, Δ +0.001 [−0.018, +0.019]). The gain is specific to our score.
>
> **The gap to Lunit closes to parity on this cohort.** Raw vs raw we are clearly behind
> (−0.054 [−0.094, −0.015]); fused-ours vs raw-Lunit is **+0.004 [−0.034, +0.039]**, and
> fused vs fused +0.003 [−0.026, +0.033]. Indistinguishable — not superior.

This is the first intervention in the project that meets its preregistered minimum
effect on the screening cohort.

## 1. Results — 10-fold stratified CV, 20 repeats, out-of-fold pooled

| image score | model | **AUROC** (mean ± SD over repeats) | prev-TB (n=466, 49 pos) | no prev-TB (n=886) | AUPRC | sens @ 0.70 spec¹ | spec¹ |
|---|---|---|---|---|---|---|---|
| ours | **M0** image only | 0.7237 | 0.6144 | 0.7815 | — | 0.619 | 0.699 |
| — | M1 clinical only | 0.6512 ± 0.004 | 0.6222 | 0.6656 | — | 0.498 | 0.666 |
| ours | **M2** image + clinical | **0.7809 ± 0.002** | **0.7122** | 0.8129 | | **0.729** | 0.697 |
| ours | M3 + score×past_tb | 0.7814 ± 0.003 | 0.7101 | 0.8135 | | 0.716 | 0.697 |
| ours_B1 | M0 | 0.7441 | 0.6338 | 0.7993 | | 0.624 | 0.699 |
| ours_B1 | **M2** | **0.7925 ± 0.002** | **0.7245** | 0.8235 | | **0.752** | 0.699 |
| **lunit** | **M0** | **0.7780** | 0.7062 | 0.8162 | | 0.708 | 0.699 |
| lunit | M2 | 0.7778 ± 0.002 | 0.7144 | 0.8081 | | 0.745 | 0.697 |
| lunit | M3 | 0.7761 ± 0.003 | 0.7085 | 0.8088 | | 0.737 | 0.698 |

¹ threshold at ≥ 0.70 specificity on each fold's *training* portion, applied to the held-out
fold — mildly optimistic for specificity, which is why it lands at 0.697–0.699 rather than
exactly 0.70. Prevalence 0.1028. Full table: `tables/fusion_results.csv`.

**At the WHO screening operating point, sensitivity moves 0.619 → 0.729 with specificity
held** — 15 more of the 139 cases found. With the active-TB-target score (B1) as the image
input: 0.752, above Lunit's raw 0.708 at the same specificity.

## 2. Paired contrasts (2,000 patient resamples on repeat-averaged OOF predictions)

| contrast | Δ AUROC, all | 95% CI | prev-TB Δ | 95% CI | meets +0.02 |
|---|---|---|---|---|---|
| **ours: M2 − M0** | **+0.058** | **[+0.030, +0.085]** | **+0.100** | **[+0.035, +0.165]** | **yes** |
| ours: M3 − M0 | +0.059 | [+0.030, +0.086] | +0.098 | [+0.019, +0.176] | yes |
| ours: M3 − M2 (interaction) | +0.001 | [−0.005, +0.006] | −0.002 | [−0.019, +0.014] | no |
| ours_B1: M2 − M0 | +0.049 | [+0.024, +0.074] | +0.092 | [+0.035, +0.153] | yes |
| **lunit: M2 − M0** | **+0.001** | **[−0.018, +0.019]** | +0.009 | [−0.032, +0.049] | **no** |
| ours raw − lunit raw | −0.054 | [−0.094, −0.015] | −0.092 | [−0.174, −0.009] | — |
| **ours fused − lunit raw** | **+0.004** | **[−0.034, +0.039]** | +0.008 | [−0.066, +0.083] | — |
| ours fused − lunit fused | +0.003 | [−0.026, +0.033] | −0.001 | [−0.063, +0.063] | — |

`tables/paired_contrasts.csv`.

**The previous-TB-specific slope (M3) adds nothing** over plain fusion. A separate slope
for previous-TB patients is not needed; the main effect of `past_tb` does the work.

## 3. Why it helps us and not Lunit — the coefficients

Full-data M3 fit, log-odds, for interpretation only (`tables/coefficients_full_fit.csv`):

| feature | ours | lunit |
|---|---|---|
| image score (standardised) | 1.14 | 1.25 |
| **past_tb** | **−0.33** | −0.23 |
| **scanner 1** | **+0.71** | +0.40 |
| **scanner 2** | **+0.83** | +0.46 |
| fever | +0.76 | +0.48 |
| weight loss | +0.74 | +0.70 |
| age ≥ group 2 | −0.84 | −0.50 |
| sex | +0.63 | +0.62 |

**Our score needs twice the scanner correction Lunit's does** (0.71/0.83 vs 0.40/0.46).
That is the same scanner confound R3 found — the exploratory per-scanner offset was worth
+0.0206 on REPORT — now handled inside a model that also uses `past_tb` and symptoms.
Fever and weight loss carry genuine signal (clinical-only AUROC is 0.651).

**Honest reading:** fusion is **repairing known weaknesses in our image score** — scanner
dependence and previous-TB over-calling — and adding symptom information that Lunit's score
already appears to encode. The repaired system reaches the comparator's level. It does not
exceed it, and it does not show our image model to be better.

## 4. What this licenses, and what it does not

**Licenses:**
- Building clinical-context fusion into the serving layer as an **optional input**, with
  the image-only path retained. A screening form has these fields; a product that uses them
  is strictly more capable than one that cannot.
- Stating, exploratorily, that *with* form data the system performs at the level of the
  commercial comparator on this cohort, including in the previous-TB population.

**Does not license:**
- Any superiority claim. Every cross-system interval includes zero.
- Any claim about screening populations other than this one. TUNE selected the pooling
  configuration; it is not independent.
- Precision in the previous-TB subgroup — 49 positives; the interval is [+0.035, +0.165].
- Adoption into production. That requires a cohort this project does not have.

## 5. Limitations

- Exposed cohort (absolute numbers optimistic; increment is CV-clean).
- `past_tb` and symptoms are **self-reported**; a product would inherit that noise.
- `dm = 99` (unknown) in 194 participants, handled by an indicator.
- Training-fold thresholds for the operating point (§1 footnote).
- One LR specification, C = 1, no tuning — deliberately; a tuned model would be
  optimistic.
- The comparator's score is a fixed commercial output; its fusion weights were CV'd here
  identically, but its training exposure to this cohort is unknown.

## 6. Files

`tables/fusion_results.csv` · `tables/paired_contrasts.csv` ·
`tables/coefficients_full_fit.csv` · `tables/run_manifest.json` · `scripts/fusion_cv.py` ·
`tests/test_fusion.py` (8 tests). No per-patient output.
