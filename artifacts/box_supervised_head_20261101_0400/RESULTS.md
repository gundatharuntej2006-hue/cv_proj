# Box-supervised regional head — results

**Run:** `artifacts/box_supervised_head_20261101_0400/` · 2026-11-01 · 3.7 min
**Protocol sha-256:** `68406c49…`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged
**Exploratory.** Calibration is the primary split (provisional C01); Vietnam TUNE is exposed
and reported for context. Nothing adopted.

---

## Two results

> **1. Region-level supervision from TBX11K's active/obsolete lesion boxes does not improve
> on image-level relabelling. Verdict: REJECT.** R − B1 on calibration **+0.0019
> [−0.0019, +0.0061]**, half-width 0.004 — a tight null. On Vietnam TUNE it is *worse*:
> **−0.0108 [−0.0182, −0.0037]**.
>
> **2. The production head already localises.** Its arg-max region overlaps an annotated
> active lesion in **95.9% [89.8, 100]** of the 49 development images that carry one
> (top-3: 100%), against **36.7%** for a random region inside the lungs, **18.4%** for a
> random region anywhere, and a mean active-box coverage of 19.7% of regions. This is
> the first measured evidence behind "lesion localisation" in the project title — and it
> was obtained **without any box supervision**.

## 1. Correctness gate — passed

| arm | R12 calibration AUROC | here | |
|---|---|---|---|
| A | 0.9498 | **0.9498** | pass |
| B1 | 0.9460 | **0.9460** | pass |

The trainer with λ = 0 reproduces the production path to four decimals. Per-seed spread
≤ 0.0015 in every arm.

## 2. Discrimination

| arm | target | region loss | **cal AUROC vs active** | 95% CI | AUPRC | sens @ dev-0.90-spec | spec | TUNE (expl.) |
|---|---|---|---|---|---|---|---|---|
| A | `tb_label` | — | **0.9498** | [0.928, 0.969] | 0.688 | 0.852 | 0.916 | 0.7237 |
| B1 | `active_tb_label` | — | 0.9460 | [0.923, 0.965] | 0.669 | 0.864 | 0.907 | **0.7441** |
| **R** | `active_tb_label` | **λ = 1** | 0.9479 | [0.927, 0.966] | 0.679 | 0.852 | 0.908 | 0.7333 |

Paired, 4,000 shared-index resamples:

| contrast | cal Δ | 95% CI | TUNE Δ | 95% CI |
|---|---|---|---|---|
| **R − B1 (primary)** | **+0.0019** | **[−0.0019, +0.0061]** | **−0.0108** | **[−0.0182, −0.0037]** |
| R − A | −0.0019 | [−0.0059, +0.0023] | +0.0095 | [+0.0036, +0.0158] |
| B1 − A | −0.0037 | [−0.0081, +0.0002] | +0.0203 | [+0.0127, +0.0279] |

Region supervision lands R between A and B1 on both splits: it recovers about half of
B1's external gain and none of its (small) internal cost. The spatial constraint adds
nothing the image-level relabel does not already carry, and on the screening cohort it
subtracts.

**Why it plausibly fails:** 62,637 regions in TB images were masked as "location unknown"
and 391,370 negatives came almost entirely from healthy images, so the region loss mostly
re-teaches "healthy regions are negative" — information the MIL max already has — while
the 12,697 positive regions are coarse (a box typically spans a 2×2–3×3 block of a 12×12
grid). λ was fixed at 1 by protocol and not tuned; a tuned λ could only be chosen on an
evaluation split, so it was not tried.

## 3. Localisation — development images with ≥ 1 active box, n = 49

| | top-1 hit | 95% CI | top-3 hit |
|---|---|---|---|
| **A (production)** | **0.959** | [0.898, 1.000] | 1.000 |
| B1 | 0.959 | [0.898, 1.000] | 1.000 |
| R (box-supervised) | 0.939 | [0.857, 1.000] | 1.000 |
| baseline: random lung region | 0.367 | [0.245, 0.510] | 0.714 |
| baseline: random region | 0.184 | [0.082, 0.306] | 0.469 |
| baseline: centre region | 0.082 | [0.020, 0.163] | 0.306 |
| chance: mean active-box coverage | 0.197 of regions | | |

Attribution here is exact: the image logit is the max over regions, so the top-1 region
*is* the decision. **The decision region lands on a real active lesion 2.6× as often as a
random lung region and 4.9× as often as chance.** Box supervision does not improve it
(0.939 vs 0.959 — within noise).

Caveats, stated plainly: 49 images; a curated benchmark whose boxes are large relative to
the grid (19.7% mean coverage, which is why top-3 saturates at 100%); these images were
used for image-level early stopping (never for the box loss, and development boxes were
never read in training). A hit means the region overlaps the box by at least one patch —
not that the model outlined the lesion. **This is "the strongest region is on a lesion",
not a detector.** Exploratory, on TBX11K only; Vietnam has no boxes.

## 4. What changes

- The thesis clause "exact region-level attribution" now has a **measured** result
  behind it, not just a mathematical one.
- Box-level supervision is **closed** as a methodology on this data — tested, null
  internally, negative externally.
- The active-TB relabel (B1) remains the only training-side change with an external
  effect, and it remains **not adopted** (R12 verdict stands).

## 5. Files

`tables/reproduction_gate.csv` · `metrics_by_arm.csv` · `paired_differences.csv` ·
`per_seed.csv` · `localisation.csv` · `geometry_and_labels.json` (bins, label counts) ·
`run_manifest.json` · `scripts/box_supervised.py` · `tests/`. No per-case output. REPORT
and `internal_test` not referenced.
