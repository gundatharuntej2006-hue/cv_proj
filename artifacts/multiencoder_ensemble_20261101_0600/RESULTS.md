# Multi-encoder regional ensemble — results

**Run:** `artifacts/multiencoder_ensemble_20261101_0600/` · 2026-11-01 · embed 8.5 min + fit 6.7 min
**Protocol sha-256:** see `tables/protocol_hash.txt`, hashed before fitting
**Git HEAD before and after:** `f878a6c`, unchanged
**Exploratory.** Calibration is the primary split (provisional C01); Vietnam TUNE is exposed
and reported for context. Nothing adopted.

---

## The result

> **Averaging three frozen CXR encoders does not beat RAD-DINO alone. Verdict: REJECT.**
> E3 − D on calibration **+0.0006 [−0.0067, +0.0076]**; on Vietnam TUNE +0.0109
> [−0.0002, +0.0215].
>
> **The secondary result is the one that matters.** Swapping the encoder for
> **RAD-DINO-MAIRA-2** (same architecture, larger self-supervised CXR corpus) — one line
> of code, no change to head, trainer, inputs or cost — is **−0.0004 [−0.0076, +0.0067]**
> on calibration and **+0.0188 [+0.0015, +0.0364]** on the screening cohort. That is
> the same size as the active-TB relabel (R12, +0.0203) and, like it, costs nothing
> internally. XraySigLIP is weaker on both splits and drags every ensemble it enters.

## 1. Correctness gate — passed

D (RAD-DINO, existing cache, same trainer): calibration **0.9498**, expected 0.9498.

## 2. Encoders — cost

| encoder | params | grid | encode s/img (GPU fp16) | peak GPU |
|---|---|---|---|---|
| D RAD-DINO | 86.6 M | 37×37 | 0.030 (R15) | 0.69 GB |
| D2 RAD-DINO-MAIRA-2 | 86.6 M | 37×37 | 0.044–0.054 | 0.69 GB |
| S XraySigLIP vision tower | 93.5 M | 32×32 | 0.035 | 0.53 GB |

(D2 timings include the cold start of the first set; same architecture as D, so the
marginal cost of a swap is zero.)

## 3. Discrimination

| arm | dev AUROC | **cal AUROC vs active** | 95% CI | TUNE (expl.) | TUNE CI |
|---|---|---|---|---|---|
| D | 0.9368 | **0.9498** | [0.928, 0.969] | 0.7237 | [0.685, 0.763] |
| **D2** | 0.9364 | 0.9493 | [0.927, 0.967] | **0.7426** | [0.705, 0.780] |
| S | 0.9361 | 0.9270 | [0.891, 0.956] | 0.7067 | [0.668, 0.746] |
| E3 (D+D2+S) | 0.9427 | 0.9503 | [0.927, 0.969] | 0.7347 | [0.697, 0.772] |
| E_DS | 0.9423 | 0.9472 | [0.922, 0.968] | 0.7252 | [0.687, 0.763] |
| E_DD2 | 0.9397 | **0.9517** | [0.930, 0.970] | 0.7368 | [0.699, 0.775] |
| E_D2S | 0.9418 | 0.9462 | [0.921, 0.967] | 0.7349 | [0.698, 0.772] |
| C3 (concat 2304-d) | **0.9505** | 0.9504 | [0.926, 0.970] | 0.7101 | [0.672, 0.748] |

Paired against D, 4,000 shared-index resamples:

| contrast | cal Δ | 95% CI | TUNE Δ | 95% CI | |
|---|---|---|---|---|---|
| **E3 − D (primary)** | **+0.0006** | **[−0.0067, +0.0076]** | +0.0109 | [−0.0002, +0.0215] | **REJECT** |
| **D2 − D** | −0.0004 | [−0.0076, +0.0067] | **+0.0188** | **[+0.0015, +0.0364]** | hypothesis |
| S − D | −0.0227 | [−0.0502, −0.0007] | −0.0171 | [−0.0442, +0.0084] | worse |
| E_DS − D | −0.0026 | [−0.0126, +0.0063] | +0.0015 | [−0.0111, +0.0134] | |
| E_DD2 − D | +0.0019 | [−0.0020, +0.0061] | +0.0131 | [+0.0043, +0.0220] | |
| E_D2S − D | −0.0035 | [−0.0154, +0.0073] | +0.0111 | [−0.0062, +0.0278] | |
| C3 − D | +0.0007 | [−0.0120, +0.0110] | −0.0137 | [−0.0324, +0.0041] | |

**Readings.**
- The concatenated head (C3) has the best development AUROC of anything ever fitted
  (0.9505) and the worst external transfer among the D-containing arms: 2,304 inputs on
  3,241 images, early-stopped on development, is enough extra capacity to fit the source.
  The pattern "more internal, less external" (record §5) holds again.
- Ensembling with a weaker encoder (S) is a regression toward it; ensembling D with D2
  is a weaker version of D2 alone on TUNE. Averaging gives nothing that choosing the
  better encoder does not.
- D2's gain on TUNE with no change on calibration is the signature already seen for the
  active-TB relabel and for multi-source training: the three known external levers are
  all "invisible internally, positive externally", and they are **independent choices**
  (encoder, label, sources).

## 4. What changes

- **Encoder ensembling is closed**: a 0.0006 null on the primary split, with the full
  cost of three encoders.
- **New pre-registered hypothesis for a next run (not a result here):** the three
  external levers stacked — RAD-DINO-MAIRA-2 encoder, active-TB target, three training
  sources. If additive (+0.019 + 0.020 + 0.008 on TUNE) the head would sit near 0.77,
  within noise of the comparator's raw 0.7862, before clinical fusion. Each lever was
  selected on the exposed TUNE cohort; the stack therefore inherits that exposure and
  **cannot be validated on TUNE** — only reported on calibration (primary) and TUNE
  (context), with REPORT remaining sealed.
- Production is unchanged. RAD-DINO `110cbc18…` remains the served encoder.

## 5. Files

`tables/encode_timing.csv` · `reproduction_gate.csv` · `metrics_by_arm.csv` ·
`paired_differences.csv` · `per_seed.csv` · `standardisation.json` · `run_manifest.json`
· `protocol_hash.txt` · `cache/` (regional features, 2.4 GB, D2 and S, four sets each;
`lungs512_calibration/` masked PNGs built with the R12 rule) · `scripts/ensemble.py` ·
`tests/` (10 pass). No per-case output. REPORT and `internal_test` not referenced or
embedded.
