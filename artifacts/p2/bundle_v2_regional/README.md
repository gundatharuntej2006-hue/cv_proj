# bundle_v2_regional — the regional head, served

**Built:** 2026-11-01 by `scripts/p2_37_build_bundle_v2.py` · head sha-256 `03546031…`
**Why it exists:** `artifacts/p2/bundle/` (v1) is the 2026-10-14 global head
(`radino_full_clahe0_bin`, Vietnam full-cohort 0.6744). Every result since 2026-10-20 is
about the regional 12×12 max head (Vietnam REPORT 0.7242). The served product was two
model generations behind the research. This bundle closes that gap without touching v1.

## Contents

| file | what |
|---|---|
| `head.npz` | `W [5,1,768]`, `b [5,1]` — the five production seeds (17/29/43/61/79). Score = mean over seeds of max over 144 regions. 3,845 trainable parameters on a frozen 86.6 M backbone |
| `bundle.json` | provenance, dev AUROC 0.9368, thresholds at 0.70 / 0.90 specificity **fitted on TBX11K development only** (−3.3368 / −0.9052), known external figures |
| `fusion.json` | logistic-regression coefficients for clinical-context fusion — **EXPLORATORY**, fitted on Vietnam TUNE (exposed). Optional at inference; refused if any field is missing |

## Runtime

`src/p2_cls/pipeline_v2.py` — standardise → quality gate → **lung mask** (torchxrayvision
PSPNet via `p1_build_c04_c05.lung_mask`, outside-lung pixels set to image mean, exactly as
the training cache was built) → regional embed (`p2_31_regional_mil.RegionalEmbedder`) →
head → operating points → optional fusion.

`scripts/serve_v2.py` — the v1 API plus `/v2/analyse`, `/v2/model`, `/v2/schema/decision`.
No edit to `src/p2_serve/app.py`.

## Verified (`scripts/p2_38_smoke_v2.py`, 2026-11-01)

Runtime mask→embed→head on raw 512 px PNGs vs the offline cached regional score, same head:

| set | n | max abs diff | median |
|---|---|---|---|
| TBX11K development (4 active TB, 4 healthy) | 8 | 0.0080 | — |
| Vietnam TUNE (with clinical fusion applied) | 8 | 0.0017 | — |
| **all** | **16** | **0.0080** | **0.0009** |

So live segmentation + embedding reproduces the training-time features. Latency: ~41 s
first call (model loads), then ~2.3 s/image on an RTX 4050 laptop — the segmenter is now
the cost, not the encoder (0.03 s).

## What a response carries that v1 could not

- `region_logits` — the 12×12 per-region logit map. The image logit **is** its max, so
  `argmax_region` is the decision region exactly. Attribution by construction.
- `operating_points` — `screen_positive_at_spec_0.70` (WHO screening point) and
  `positive_at_spec_0.90`, both with their development-fitted thresholds stated.
- `fusion.p_fused` — only when the full screening form is supplied, always labelled
  EXPLORATORY with its source run. CV result on TUNE: 0.7237 → 0.7809 overall,
  0.6144 → 0.7122 in previous TB (`artifacts/clinical_fusion_20261101_0330`).

## What it does not do, deliberately

No OOD gate (the v1 bank is for the global embedder). No temperature scaling (none fitted
for this head). No `TB` / `NOT_TB` label — only `screen_positive` at a stated operating
point. **Not clinically validated. Provisional. Split is the provisional C01.**

Vietnam REPORT and TBX11K `internal_test` were not read for any weight, threshold or
coefficient in this bundle.
