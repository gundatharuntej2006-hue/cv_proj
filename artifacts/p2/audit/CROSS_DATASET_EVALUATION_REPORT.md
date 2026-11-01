# Cross-dataset paired pooling comparison: Shenzhen and Montgomery

**Owner:** Person 2 · **Date:** 2026-10-21 · **Repo HEAD at run:** `23febae`
**Protocol:** `cross_dataset_manifest.json`, sha256 `610af8f0330af697…`, committed before
any feature here was computed.

> **CLASSIFICATION: EXPLORATORY PAIRED BENCHMARK. NOT INDEPENDENT EXTERNAL VALIDATION.**
> The lungs-only masking in the configuration under test was adopted partly because of how
> it scored on these two cohorts. Only the *within-pair difference* between aggregators is
> interpreted. No absolute number here is a clean generalisation estimate.

## Setup

Three arms, all fitted on **TBX11K train only** (3,241 cases, 541 TB, healthy-only
negatives), early-stopped on TBX11K development (659), seeds (17, 29, 43, 61, 79). **No arm
ever saw a Shenzhen or Montgomery label.** Thresholds fixed at 70% specificity on TBX11K
development and transferred unchanged.

| arm | representation | aggregation | TBX11K dev AUC |
|---|---|---|---|
| **A** global_pooler | RAD-DINO `pooler_output` | none | 0.9570 |
| **B** patch_mean_1×1 | 12×12 regions → 1×1 | mean | 0.8391 |
| **C** regional_max_12×12 | 12×12 regions | max | 0.9368 |

**Eligibility:** 800 of 800 NLM images produced a lung mask and embedded successfully in
both arms. **Zero exclusions, zero regional-only or global-only cases, no duplicate ids.**
Arms are scored on an identical case set.

## Results

### Shenzhen — n=662, 336 TB, prevalence 0.508

| arm | AUROC | 95% CI | AUPRC | sens | spec |
|---|---|---|---|---|---|
| A global_pooler | 0.9411 | [0.9233, 0.9570] | 0.9540 | 0.887 | 0.791 |
| B patch_mean_1×1 | 0.8471 | [0.8177, 0.8752] | 0.8689 | 0.774 | 0.752 |
| **C regional_max_12×12** | **0.9690** | [0.9551, 0.9809] | 0.9765 | 0.887 | 0.954 |

| paired difference | ΔAUROC | 95% CI | verdict |
|---|---|---|---|
| C − B (pure pooling) | **+0.1220** | [+0.0938, +0.1501] | better |
| C − A (pooling + representation) | **+0.0279** | [+0.0128, +0.0434] | better |

### Montgomery — n=138, 58 TB, prevalence 0.420

| arm | AUROC | 95% CI | AUPRC | sens | spec |
|---|---|---|---|---|---|
| A global_pooler | 0.9289 | [0.8793, 0.9691] | 0.9230 | 0.759 | 0.938 |
| B patch_mean_1×1 | 0.8772 | [0.8182, 0.9300] | 0.8510 | 0.897 | 0.688 |
| **C regional_max_12×12** | **0.9690** | [0.9296, 0.9961] | 0.9744 | 0.931 | 0.938 |

| paired difference | ΔAUROC | 95% CI | verdict |
|---|---|---|---|
| C − B (pure pooling) | **+0.0918** | [+0.0375, +0.1503] | better |
| C − A (pooling + representation) | **+0.0401** | [+0.0089, +0.0767] | better |

Montgomery's intervals are wide — 58 positives — and should not be read as precisely as
Shenzhen's.

## The nuance that matters, stated before the headline is quoted

**The two paired comparisons mean different things, and the larger one is the less useful.**

- **C − B = +0.12** isolates the aggregator exactly: same tokens, same training, same seeds,
  one difference. It is the clean *scientific* comparison.
- **But B is not the system we had.** Averaging patch tokens (0.8471) is substantially worse
  than RAD-DINO's own `pooler_output` (0.9411). B is a weak baseline, not the deployed one.
- **C − A = +0.028 to +0.040** is the *practically relevant* number: what focal pooling buys
  over the representation actually in use.

Quoting "+0.12 from focal pooling" as an improvement over our system would be misleading.
The honest statement is: **focal pooling improves on the deployed global representation by
0.028–0.040 AUROC at these two sites, both intervals excluding zero.**

## What these results do and do not support

**Supported:**
- Within these cohorts, under identical preprocessing, training data and seeds, regional
  max-pooling scored higher than both the matched mean control and the deployed global
  representation, at both sites, with all four paired intervals excluding zero.
- The direction is now consistent across four cohorts (TBX11K dev, Vietnam, Shenzhen,
  Montgomery), which was previously untested outside Vietnam.

**Not supported, and explicitly disclaimed:**
- *Independent external validation.* These cohorts selected part of the configuration.
- *Evidence of generalisation.* See above, plus the RAD-DINO checkpoint-selection history,
  which the model card leaves unresolvable for any public benchmark.
- *Novelty.* RADAR and GLoRI already cover patch-level aggregation over frozen CXR
  foundation models (`PRIOR_ART_AND_NOVELTY.md`).
- *Clinical readiness.* Nothing here is prospective.

**Context that stops these numbers being over-read:** Shenzhen and Montgomery are balanced
case-control sets (prevalence 0.51 and 0.42). The Vietnam screening cohort has prevalence
0.103 and the same configuration scores **0.7242** there. The 0.969 figures reflect how much
easier a curated balanced benchmark is, not an improvement in the system. This is the same
ladder documented in `p2_vietnam_result.md` and these results sit at its easy end.

## Process note: an error caught before it produced a number

The first extraction wrote `(800, 324, 768)` — an 18×18 grid, because `p2_31`'s `GRID`
constant was still set from the earlier sweep, while TBX11K's cached features are 12×12
(144 regions). The head is a per-region linear layer on 768-d vectors, so **this would not
have raised an error**: it would have trained max-over-144 and evaluated max-over-324,
silently changing the configuration between fit and scoring.

It surfaced only because the array shape was checked against the grid the manifest
specifies. Features were re-extracted at 12×12 and the mislabelled file deleted. This is a
direct argument for writing the manifest first with concrete values: it provided something
falsifiable to check against.

## Reproduction

```
py -3 scripts/p2_36_cross_dataset_pooling.py --stage embed   # ~1 min, 800 images
py -3 scripts/p2_36_cross_dataset_pooling.py --stage score   # ~3 min
```

Outputs: `cross_dataset_metrics.csv`, `cross_dataset_case_predictions.csv` (800 rows,
dataset IDs only), `cross_dataset_run.json`.

## Conclusion these results support, in one sentence

> Under a pre-registered protocol with identical training data, preprocessing and seeds,
> regional max-pooling over 12×12 patch regions scored 0.028–0.040 AUROC above the deployed
> global representation on Shenzhen and Montgomery, with paired intervals excluding zero —
> an exploratory within-cohort comparison on benchmarks that previously informed part of
> this configuration, and therefore not an independent estimate of generalisation.
