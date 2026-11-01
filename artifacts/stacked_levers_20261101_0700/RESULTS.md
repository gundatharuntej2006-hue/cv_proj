# The three external levers, stacked — results

**Run:** `artifacts/stacked_levers_20261101_0700/` · 2026-11-01 · 7.3 min
**Protocol sha-256:** see `tables/protocol_hash.txt`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged
**Exploratory.** Calibration is the primary split (provisional C01); Vietnam TUNE is exposed
— and is where every lever was chosen. Nothing adopted.

---

## The result

> **On the primary split the stack is a null, as every external lever has been:
> X − A on calibration −0.0010 [−0.0099, +0.0083]. Verdict: REJECT.**
>
> **On the screening cohort the levers are not additive.** Predicted if additive: +0.047.
> Observed: X − A **+0.0286 [+0.0089, +0.0490]**. The encoder and the label gains
> overlap almost entirely; the three sources add nothing once the encoder is MAIRA-2
> (X − X_noNLM −0.0062 [−0.0124, +0.0001]). The best head on TUNE is the ablation
> **X_noNLM — MAIRA-2 + active target, TBX11K only: 0.7585 [0.722, 0.796]**,
> +0.0348 [+0.0160, +0.0542] over production, −0.0032 on calibration.

## 1. Correctness gates — passed

| arm | expected | here |
|---|---|---|
| A (RAD-DINO, tb) | 0.9498 | **0.9498** |
| MB1 (RAD-DINO, active, + NLM; R19) | 0.9497 | **0.9497** |

## 2. Discrimination

| arm | encoder | target | + NLM | dev | **cal AUROC vs active** | 95% CI | AUPRC | TUNE (expl.) | TUNE CI | TUNE AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| A | RAD-DINO | tb | – | 0.9368 | **0.9498** | [0.928, 0.969] | 0.688 | 0.7237 | [0.685, 0.763] | 0.201 |
| MB1 | RAD-DINO | active | ✓ | 0.9355 | 0.9497 | [0.929, 0.968] | 0.689 | 0.7517 | [0.713, 0.789] | 0.231 |
| **X** | MAIRA-2 | active | ✓ | 0.9362 | 0.9488 | [0.927, 0.967] | 0.660 | 0.7524 | [0.715, 0.789] | 0.238 |
| **X_noNLM** | MAIRA-2 | active | – | 0.9317 | 0.9466 | [0.925, 0.965] | 0.646 | **0.7585** | [0.722, 0.796] | **0.264** |
| X_tb | MAIRA-2 | tb | ✓ | 0.9399 | **0.9513** | [0.929, 0.969] | 0.685 | 0.7359 | [0.698, 0.774] | 0.211 |

Paired, 4,000 shared-index resamples:

| contrast | cal Δ | 95% CI | TUNE Δ | 95% CI | |
|---|---|---|---|---|---|
| **X − A (primary)** | **−0.0010** | **[−0.0099, +0.0083]** | +0.0286 | [+0.0089, +0.0490] | **REJECT** |
| X − MB1 (encoder, on top of the other two) | −0.0010 | [−0.0094, +0.0077] | +0.0006 | [−0.0171, +0.0184] | nothing |
| X_noNLM − A (encoder + label) | −0.0032 | [−0.0110, +0.0046] | **+0.0348** | **[+0.0160, +0.0542]** | best |
| X_tb − A (encoder + sources) | +0.0016 | [−0.0060, +0.0093] | +0.0121 | [−0.0058, +0.0302] | |
| X − X_noNLM (sources, given encoder + label) | +0.0022 | [−0.0010, +0.0056] | −0.0062 | [−0.0124, +0.0001] | |
| X − X_tb (label, given encoder + sources) | −0.0025 | [−0.0057, +0.0006] | +0.0165 | [+0.0088, +0.0244] | |
| MB1 − A | −0.0000 | [−0.0050, +0.0050] | +0.0280 | [+0.0172, +0.0386] | R19 |

**Additivity check (`run_manifest.json`):** sum of single-lever TUNE deltas +0.0468;
observed X − A +0.0286. The levers share a mechanism — plausibly all three push the
head away from TBX11K's particular rendering of healed disease — so the second and third
buy less than the first.

Operating points transfer exactly as before: at the development 0.70-specificity
threshold every arm screens 99–100% of TUNE positives at 12–14% specificity. The
threshold still has to be set locally (R15); the ranking is what these arms change.

## 3. What this means

- **Head-level choices on TUNE have a ceiling near 0.76**, reached by MAIRA-2 + active
  target. The comparator's raw AUROC on the same patients is 0.7862. The remaining
  0.03 is not available from the encoder, the label, the sources, their combination,
  box supervision, attention pooling, head adaptation or encoder ensembling — all
  measured, all closed.
- **The ceiling is a TUNE-designed number.** Every lever was selected on this cohort, so
  0.7585 is a best-case, not an estimate. Its CI on TUNE [0.722, 0.796] already contains
  the comparator.
- Clinical-context fusion (R16) was worth +0.058 on arm A and +0.052 on B1. Whether it
  is worth the same on X_noNLM — which would put the fused score above the comparator's
  raw score on this cohort, with the comparator gaining nothing from fusion — is the
  next measurement, by the same nested-CV protocol, with the same exposure caveat.
- Production is unchanged. Adopting MAIRA-2 + active target would be a product
  decision made on an exposed cohort; the record's rule (R12) is that this is not done
  without a sealed evaluation, and REPORT stays sealed.

## 4. Files

`tables/reproduction_gate.csv` · `metrics_by_arm.csv` · `paired_differences.csv` ·
`per_seed.csv` · `run_manifest.json` · `protocol_hash.txt` · `cache/regional_D2_nlm_lungs_g12.npz`
(MAIRA-2 regional features, 800 NLM images) · `cache/X_head.npz` (arm X weights, five seeds)
· `scripts/stacked.py` · `tests/` (7 pass). No per-case output. REPORT and
`internal_test` not referenced or embedded.
