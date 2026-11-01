# Multi-source regional head — results

**Run:** `artifacts/multisource_head_20261101_0500/` · 2026-11-01 · 4.5 min
**Protocol sha-256:** see `tables/protocol_hash.txt`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged
**Exploratory.** Calibration is the primary split (provisional C01); Vietnam TUNE is exposed
and reported for context. Nothing adopted.

---

## The result

> **Adding Shenzhen + Montgomery (800 images, two further scanner families) to the head's
> training set helps, measurably, and by too little to matter. Verdict: REJECT.**
> M − A on calibration **+0.0033 [+0.0006, +0.0064]** — the interval excludes zero, the
> effect is one sixth of the +0.02 bar. On Vietnam TUNE **+0.0036 [−0.0003, +0.0075]**.
> The same pattern under the active target: MB1 − B1 **+0.0037** calibration,
> **+0.0077 [+0.0016, +0.0138]** TUNE.

So the external gap is **not** a source-diversity gap at the head. Three sources instead of
one shift the head by a few thousandths; the 0.95 → 0.72 drop is the cohort (screening,
Xpert reference, previous-TB prevalence), as the fusion run (R16) already indicated.

## 1. Correctness gate — passed

| arm | expected | here | |
|---|---|---|---|
| A | 0.9498 | **0.9498** | pass |
| B1 | 0.9460 | **0.9460** | pass |

Per-seed calibration spread ≤ 0.0016 in every arm; early-stopping development AUROC
spread ≤ 0.0018.

## 2. Discrimination

| arm | training rows | target | **cal AUROC vs active** | 95% CI | AUPRC | sens @ dev-0.90-spec | spec | TUNE (expl.) | TUNE CI |
|---|---|---|---|---|---|---|---|---|---|
| A | 3,241 TBX11K | tb | **0.9498** | [0.928, 0.969] | 0.688 | 0.852 | 0.897 | 0.7237 | [0.685, 0.763] |
| **M** | + 800 NLM | tb | **0.9531** | [0.932, 0.971] | 0.704 | 0.864 | 0.902 | 0.7274 | [0.688, 0.767] |
| B1 | 3,241 TBX11K | active | 0.9460 | [0.923, 0.965] | 0.669 | 0.864 | 0.890 | 0.7441 | [0.705, 0.783] |
| **MB1** | + 800 NLM | active | 0.9497 | [0.929, 0.968] | 0.689 | 0.864 | 0.886 | **0.7517** | [0.713, 0.789] |

Paired, 4,000 shared-index resamples:

| contrast | cal Δ | 95% CI | TUNE Δ | 95% CI | |
|---|---|---|---|---|---|
| **M − A (primary)** | **+0.0033** | **[+0.0006, +0.0064]** | +0.0036 | [−0.0003, +0.0075] | **REJECT** (< 0.02) |
| MB1 − B1 | +0.0037 | [−0.0003, +0.0085] | +0.0077 | [+0.0016, +0.0138] | |
| MB1 − A | −0.0000 | [−0.0050, +0.0050] | **+0.0280** | [+0.0172, +0.0386] | |
| B1 − A | −0.0037 | [−0.0081, +0.0002] | +0.0203 | [+0.0127, +0.0279] | |

MB1 is the best single head ever measured on TUNE (0.7517), and it reaches that with
**no** cost on calibration (MB1 − A = 0.0000). But its TUNE gain decomposes as
+0.020 from the active-TB relabel (already known, R12) and +0.008 from the extra sources —
the second part is the new information here, and it is small.

The development-fitted 0.90-specificity threshold transfers to TUNE at specificity
0.34–0.36 in every arm: the threshold, not the ranking, is what moves between sites
(R15 stands).

## 3. What changes

- **Multi-source head training is closed**: a tight, positive, sub-threshold effect on
  both splits. If NLM is ever added, it is for the +0.004–0.008, stated as such — not as
  a route to the external gap.
- R15's Shenzhen / Montgomery evaluations (0.9681 / 0.9483) remain valid for arm A only;
  arms M / MB1 have seen those sites.
- The combination MB1 (active target + three sources) is the strongest *screening-cohort*
  head on record and remains **not adopted**: its primary-split contrast is exactly zero
  and the active-target decision (R12) is unchanged.
- Assumption carried: every NLM TB image was treated as active under the active target.
  Shenzhen / Montgomery do not label obsolete disease; the +0.0037 may be deflated by
  obsolete cases labelled positive.

## 4. Files

`tables/training_sets.json` · `reproduction_gate.csv` · `metrics_by_arm.csv` ·
`paired_differences.csv` · `per_seed.csv` · `run_manifest.json` · `protocol_hash.txt` ·
`scripts/multisource.py` · `tests/`. No per-case output. REPORT and `internal_test`
not referenced.
