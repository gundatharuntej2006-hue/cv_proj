# Clinical-context fusion on the MAIRA-2 + active-target head — results

**Run:** `artifacts/fusion_on_stacked_head_20261101_0800/` · 2026-11-01 · 3.4 min
**Protocol sha-256:** see `tables/protocol_hash.txt`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged
**Exploratory, exposed.** Everything is on Vietnam TUNE by nested CV; the head's encoder
and target were chosen on this cohort (R21). Nothing adopted.

---

## The result

> **The head-level gain survives fusion only in part, and not at the bar.** Fused X
> minus fused production: **+0.0133 [−0.0019, +0.0296]**. Verdict on the primary:
> **REJECT** (below +0.02, interval touches zero).
>
> **Fused X is the first score in the project that is numerically above the
> comparator's raw score on this cohort: 0.7941 vs 0.7780, +0.0170 [−0.0193, +0.0527]**
> — a point estimate above, an interval that includes zero. In previous TB,
> 0.7349 vs 0.7062 (+0.030 [−0.046, +0.107]). This is "not below the comparator", shown
> twice (R16 at parity, here above); it is not "better than the comparator".

## 1. Gates — all passed

| check | expected | here |
|---|---|---|
| X_noNLM raw TUNE AUROC (R21) | 0.7585 | **0.7585** |
| X_noNLM calibration AUROC (R21) | 0.9466 | **0.9466** |
| ours M2 (R16) | 0.7809 | **0.7809** |
| lunit M2 (R16) | 0.7778 | **0.7778** |

## 2. Results — 10-fold × 20 repeats, OOF averaged

| score | model | AUROC | ± | previous TB (466) | no previous TB (886) | AUPRC | sens @ 0.70 spec | spec |
|---|---|---|---|---|---|---|---|---|
| ours (A) | M0 raw | 0.7237 | | 0.6144 | 0.7815 | 0.201 | 0.619 | 0.699 |
| ours (A) | M2 fused | 0.7809 | 0.0021 | 0.7122 | 0.8129 | 0.276 | 0.729 | 0.697 |
| **ours_X** | M0 raw | 0.7585 | | 0.6501 | 0.8122 | 0.264 | 0.656 | 0.700 |
| **ours_X** | **M2 fused** | **0.7941** | 0.0023 | **0.7349** | **0.8224** | 0.300 | 0.741 | 0.698 |
| lunit | M0 raw | 0.7780 | | 0.7062 | 0.8162 | **0.328** | 0.708 | 0.699 |
| lunit | M2 fused | 0.7778 | 0.0019 | 0.7144 | 0.8081 | 0.306 | 0.745 | 0.697 |

Paired, 2,000 shared-index resamples:

| contrast | Δ | 95% CI | previous-TB Δ | 95% CI | |
|---|---|---|---|---|---|
| **ours_X M2 − ours M2 (primary)** | **+0.0133** | **[−0.0019, +0.0296]** | +0.0225 | [−0.0116, +0.0565] | **REJECT** |
| ours_X M2 − lunit M0 | +0.0170 | [−0.0193, +0.0527] | +0.0304 | [−0.0458, +0.1066] | above, not significant |
| ours_X M2 − lunit M2 | +0.0164 | [−0.0131, +0.0475] | +0.0215 | [−0.0423, +0.0861] | |
| ours_X M0 − ours M0 | +0.0348 | [+0.0163, +0.0543] | +0.0357 | [+0.0016, +0.0680] | R21 reproduced |
| ours_X M2 − ours_X M0 | +0.0365 | [+0.0154, +0.0576] | **+0.0865** | [+0.0349, +0.1413] | fusion still works |
| ours_X M0 − lunit M0 | −0.0195 | [−0.0562, +0.0172] | −0.0561 | [−0.1384, +0.0303] | raw: still behind |

**Readings.**
- Fusion on X is worth +0.037 (on A it was +0.058): the better head already carries
  some of what the clinical form was supplying, so the two gains overlap, as the three
  head levers overlapped in R21. Overall: raw A 0.7237 → fused X 0.7941, +0.070.
- Sensitivity at the 0.70-specificity screening point: 0.619 (raw A) → 0.741 (fused X),
  against the comparator's 0.708 raw / 0.745 fused.
- The comparator keeps the best AUPRC (0.328 vs 0.300): it ranks the top of the list
  better; ours ranks the whole cohort slightly better. Different operating regions.
- Previous TB remains the weakest subgroup for every score (0.73 at best) — the fusion
  repair is largest there (+0.087) and still leaves it 0.09 below the no-previous-TB
  group.

## 3. What this is and is not

**Is:** the strongest product configuration measured — MAIRA-2 encoder, active-TB
target, exact regional attribution, optional 12-field screening form — with a fused
AUROC of 0.794 on the only screening cohort available, numerically above the
comparator's 0.778, with the comparator unable to use the same form.

**Is not:** evidence of superiority. The cohort chose the encoder, the target and the
fusion; the superiority interval includes zero; REPORT is sealed and unexamined. The
defensible sentence is: *on an exposed screening cohort, a frozen-encoder regional head
with clinical-context fusion is not worse than the commercial comparator and is
numerically above it (+0.017 [−0.019, +0.053]).* Anything stronger needs the data
collaboration.

## 4. Files

`tables/fusion_results.csv` · `paired_contrasts.csv` · `reproduction_gate.csv` ·
`X_noNLM_head.npz` (five seeds) · `run_manifest.json` · `protocol_hash.txt` ·
`scripts/fusion_x.py` · `tests/`. No per-case output; no patient identifier. REPORT and
`internal_test` not referenced.
