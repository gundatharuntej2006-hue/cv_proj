# Efficiency and site-adaptation benchmark — results

**Run:** `artifacts/site_adaptation_benchmark_20261101_0300/` · 2026-11-01 · 7.0 min
**Protocol sha-256:** `a999e45a…`, hashed before any measurement
**Git HEAD before and after:** `f878a6c`, unchanged; `git status` identical
**Exploratory.** Shenzhen and Montgomery are curated case-control sets at ~50% prevalence,
both previously used by this project. Nothing here is validation or a superiority claim.

---

## Headline: head refit gives no benefit over threshold-only recalibration

> **0 of 30 paired comparisons met the predeclared "head refit helps" rule.** 26 were
> "no clear benefit", 4 were "harm" (specificity drops of 0.025–0.031). The largest AUROC
> gain from refitting the head on local data was **+0.0031**.
>
> **The "site-adaptive by refitting the head" clause of the proposed product thesis is not
> supported and should be removed.**

The efficiency numbers, by contrast, are real, measured, and stand on their own (§1).

## 1. Cost — measured, not extrapolated

| | **GPU** (RTX 4050 Laptop, fp16) | **CPU** (8 torch threads, fp32) |
|---|---|---|
| encoder load (cold) | 27.5 s | 0.7 s¹ |
| encode, per image | **0.030 s** (33 images/s) | **1.25 s** (0.8 images/s) |
| peak GPU memory | **0.69 GB** | — |
| head fit, 3,241 TBX11K images | **6.6 s** | 61.6 s |
| trainable parameters | **769** | 769 |
| frozen backbone parameters | 86,580,480 | 86,580,480 |
| re-encoding reproduces the cached features | MAD **0.00** | n/a (fp32) |

¹ second load in the same process; weights were already in the OS cache.

**Total adaptation time**, raw labelled images → updated head, measured per sample size:

| local images | GPU encode + fit | CPU (encode extrapolated + fit measured) |
|---|---|---|
| 25 | 6.5 s | ~1.6 min |
| 100 | 8.9 s | ~3.1 min |
| 331 | **16.2 s** | ~7.9 min |

So: **screening at 33 images/s on a laptop GPU in under 0.7 GB, and a full site
adaptation in 16 seconds — or about 8 minutes on CPU with no GPU at all.** These are the
honest product facts, and they do not depend on whether adaptation helps.

## 2. The frozen head already works on both sites — there is no headroom

A0 = the production head and its TBX11K-development threshold, no adaptation, on the
held-out evaluation halves:

| site | eval n (TB) | AUROC | sens | spec | oracle sens at 0.70 spec |
|---|---|---|---|---|---|
| Shenzhen | 331 (168) | **0.9681** [0.948, 0.985] | 0.8988 | 0.9387 | 0.9702 |
| Montgomery | 69 (29) | **0.9483** [0.871, 1.00] | 0.8966 | 0.9500 | 0.9310 |

Consistent with `artifacts/p2/audit/cross_dataset_metrics.csv` (regional head, full sites:
Shenzhen 0.969, Montgomery 0.969 — already on record, not a finding of this run).

At the home threshold the system sits at a *high-specificity* point (0.94–0.95). Moving it
to the WHO screening point (0.70 specificity) is what local threshold refit does.

## 3. Threshold-only refit reaches the oracle operating point with 25 local images

Arm A1 — frozen head, threshold refit on n local images, evaluated on the held-out half:

| site | n local | sens (mean of 3 draws, range) | spec (mean) | oracle |
|---|---|---|---|---|
| Shenzhen | 10 | 0.976 (0.970–0.988) | 0.634 | 0.9702 |
| Shenzhen | **25** | **0.970** | **0.679** | 0.9702 |
| Shenzhen | 100 | 0.968 | 0.714 | 0.9702 |
| Shenzhen | 331 | 0.970 | 0.699 | 0.9702 |
| Montgomery | 25 | 0.931 | 0.583 | 0.9310 |
| Montgomery | 69 | 0.931 | 0.675 | 0.9310 |

With 25 local images the sensitivity is already the oracle value and specificity is within
0.02 of target; by 100 it is on target. At n = 10 the threshold is noisy (specificity
0.55–0.73 across draws) — ten images are not enough to place a threshold.

**This is useful, measured, and not a differentiator.** It is exactly what current
deployment guidance for commercial CAD already requires (local threshold calibration by
population and equipment).

## 4. Head refit versus threshold-only — the test of the thesis

Arm A2 — head refit on TBX11K train + n local (pooled, unweighted, early-stopped on TBX11K
development, production recipe), threshold on the same n local — against A1 at the same n,
paired on the same evaluation images:

| site | n | Δ sens (A2−A1) | Δ spec | Δ AUROC | verdicts (3 draws) |
|---|---|---|---|---|---|
| Shenzhen | 10 | 0.000 | −0.018 … 0.000 | ±0.000 | 3 × no benefit |
| Shenzhen | 25 | 0.000 | −0.012 … +0.031 | ±0.000 | 3 × no benefit |
| Shenzhen | 50 | 0.000 | −0.012 … +0.012 | +0.001 | 3 × no benefit |
| Shenzhen | 100 | −0.006 … +0.006 | −0.031 … +0.025 | +0.002 | 2 no benefit, **1 harm** |
| Shenzhen | 200 | 0.000 | −0.025 … +0.012 | +0.002 | 2 no benefit, **1 harm** |
| Shenzhen | 331 | 0.000 | **+0.031** | **+0.003** | 3 × no benefit |
| Montgomery | 10–69 | 0.000 | −0.025 … +0.025 | +0.001 … +0.003 | 10 no benefit, **2 harm** |

Sensitivity is **identical** between arms in 28 of 30 comparisons (paired CI [0, 0]) — at
the 0.70-specificity operating point both heads flag the same positives. Specificity moves
by at most ±0.03 in either direction, i.e. 1–5 patients out of 163, with no consistent
sign. AUROC moves by at most +0.003.

**Why it doesn't help — three reasons, all of which are the point:**

1. **No headroom.** The frozen head is already at 0.95–0.97 AUROC on both sites, and
   threshold refit already reaches the oracle sensitivity. There is nothing left for the
   head to fix.
2. **Conservative design, by choice.** Local images are ≤ 9% of the pooled training set,
   unweighted, and the fit is early-stopped on TBX11K development. Upweighting local data
   or early-stopping on it would be hyperparameters tunable only on the evaluation site,
   so the protocol forbade them. A more aggressive adaptation might behave differently —
   but it could not be tested honestly here.
3. **The sites where adaptation would matter are the ones we cannot use.** Head adaptation
   is interesting where the frozen head *fails* — Vietnam, at 0.72. Vietnam is exposed.
   This benchmark tested adaptation where it was not needed, because those are the only
   sites left.

## 5. What this does to the product thesis

| clause | status after this run |
|---|---|
| computationally efficient; runs on a laptop GPU or a CPU | **measured and supported** — §1 |
| 769 trainable parameters on a frozen public backbone | **supported**, with the frozen 86.6 M stated alongside |
| locally adaptable: threshold recalibration from ~25 local images | **supported**, and it is parity with commercial practice, not an advantage |
| **locally adaptable: head refit improves on threshold-only** | **NOT supported. Remove.** |
| exact region-level attribution | unchanged — a mathematical property, still untested against lesion boxes |
| better on the previous-TB population | unchanged — needs data that does not exist here |

The defensible version:

> A computationally efficient TB-screening research system — 33 images/s and 0.7 GB on a
> laptop GPU, 0.8 images/s on CPU — built on a frozen public foundation model with a
> 769-parameter head whose decision region is exactly identifiable, and whose operating
> point can be recalibrated to a new site from ~25 labelled images in seconds.

Nothing in that sentence claims an accuracy advantage, and nothing in this run supplies
one.

## 6. Limitations

- Curated case-control sites, ~50% prevalence, previously used for external validation:
  **exploratory**, cannot estimate screening-population performance or generalisation.
- 3 draws per sample size; single head seed (17); n ≤ 331. Montgomery's 69-image
  evaluation half gives intervals like [0.87, 1.00] and should be read as a direction only.
- No local-only head arm and no upweighting — deliberately (§4.2).
- The CPU encode rate is from 32 images and is a cost estimate, not a benchmark.
- The threshold rule (≥ 0.70 specificity on the fitting images) is one operating point;
  other targets were not evaluated.

## 7. Files

`tables/timing.csv` · `tables/adaptation_curve.csv` (every arm, n, draw, with CIs) ·
`tables/head_refit_vs_threshold_only.csv` (paired deltas and verdicts) ·
`tables/adaptation_summary.csv` · `tables/site_splits.json` · `tables/run_manifest.json` ·
`scripts/benchmark.py` · `tests/test_benchmark.py` (11 tests, all passing).

No per-patient output. Vietnam REPORT and TBX11K `internal_test` not referenced. No
training of the backbone. No production file changed.
