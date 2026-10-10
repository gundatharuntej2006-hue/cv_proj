# Explainability (Grad-CAM on RAD-DINO) and the guide review

*Owner: P3. Reviewer: P4. Heatmaps are **attention evidence, not a finding**. Agreement with radiologist boxes is
overlap with incomplete annotations. It is **not** proof that an explanation is clinically correct (methodology
section 10).*

## Adapting Grad-CAM to a Vision Transformer

Grad-CAM was designed for convolutional networks, which produce a 2-D feature map. RAD-DINO is a Vision
Transformer: each block outputs a list of tokens, namely one CLS token, possibly some register tokens, and one
token per image patch. We adapted it in three steps [src/p3_xai/gradcam.py; decisions/p3_gradcam_config.json, D19]:

1. **Target layer.** We use the first LayerNorm (`norm1`, the attention input) of the last transformer block.
   We do not use the block's output. The classifier head reads the CLS token through a final per-token
   LayerNorm, so at the block output the patch tokens receive exactly zero gradient. This is verified by a test
   [tests/p3_xai/test_gradcam.py]. The layer is found by inspecting the classifier module, not hard-coded.
2. **Reshape rule.** We drop the CLS and register tokens; their number is read from the model configuration. The
   remaining patch tokens are reshaped into the patch grid (input size ÷ patch size). A mismatch in the token
   count is an error.
3. **Target and mapping.** The target is the TB logit. The map is computed on the lung crop, resized to the crop,
   normalised to [0, 1] inside it, and pasted into a 512 × 512 canvas that is zero outside the crop.

Peak GPU memory for one heatmap must be at most 5.5 GB on the RTX 4050 [decisions/p3_gradcam_config.json,
criterion]. Measured value: TBD: waiting on artifacts/p3/xai/vram_explain.json.

## Model-randomisation sanity check

A heatmap that does not change when the model's learned weights are destroyed is explaining the image, not the
model (Adebayo et al., 2018). Pre-registered check [decisions/p3_gradcam_config.json, criterion]:
- re-initialise the classifier head;
- recompute heatmaps on 20 development images (10 TB, 10 non-TB) [decisions/p3_gradcam_config.json];
- require a mean absolute Spearman correlation (|rho|) with the original heatmaps below 0.5, computed inside the
  crop only [decisions/p3_gradcam_config.json];
- report the maximum |rho| and the per-image values, and the same check with the head and last block re-initialised.

| Randomised part | Mean abs Spearman (rule) | Max abs Spearman | Mean signed Spearman (not the rule) | n images | Pass (below the pre-registered limit) | Source |
|---|---|---|---|---|---|---|
| Head | TBD | TBD | TBD | TBD | TBD | TBD: waiting on artifacts/p3/xai/sanity/randomisation_<model_id>.json |
| Head + last block | TBD | TBD | TBD | TBD | (reported) | TBD: waiting on artifacts/p3/xai/sanity/randomisation_<model_id>.json |

Why the absolute value: a signed correlation below 0.5 is also met by an *inverted* map, which still depends on the
weights [decisions/p3_gradcam_config.json, criterion]. Runs on P2's stub or on stand-in models are labelled
"stub, not evidence" and are never cited here [src/p3_xai/sanity.py].

## Localisation agreement (development)

The measures are computed on development TB images that have radiologist boxes [src/p3_xai/metrics.py]
(per-image tables: TBD: waiting on the C11 development tables from python -m p3_xai.export_c11):
- **Pointing accuracy:** the share of images whose heatmap peak lies inside any box.
- **Energy-in-box:** the share of heatmap mass inside the boxes.

CIs are percentile bootstrap over images. The development split has only about 60 TB images
[docs/methodology.md, section 1.2], so these intervals are wide.

| Model | Pointing accuracy (95% CI) | Mean energy-in-box (95% CI) | n images / boxes | Source |
|---|---|---|---|---|
| Primary (lung crop) | TBD | TBD | TBD | TBD: waiting on artifacts/p3/xai/loc_summary_development.json |
| Full-image comparator | TBD | TBD | TBD | TBD: waiting on artifacts/p3/xai/loc_summary_development.json |

## Does lung cropping help? (methodology section 14)

We compare the two models on the same images, with paired bootstrap CIs of the difference (crop − full)
[src/p3_xai/export_c11.py].

| Difference (crop − full) | Value | 95% CI | n images | Source |
|---|---|---|---|---|
| Pointing accuracy | TBD | TBD | TBD | TBD: waiting on artifacts/p3/xai/loc_summary_development.json |
| Mean energy-in-box | TBD | TBD | TBD | TBD: waiting on artifacts/p3/xai/loc_summary_development.json |

Sealed (internal-test) localisation for section 12.5: TBD: waiting on
artifacts/p3/xai/loc_<model_id>_internal_test_v1.parquet and P4's C18 table t12_5.

## Grad-CAM review with the guide

We reviewed 40 development cases with the guide: 20 classified correctly and 20 incorrectly, 10 per cell (true
and false positives, true and false negatives) [docs/p3_gradcam_review.md, section 1]. They were drawn with a pre-registered seeded rule, without
looking at the heatmaps, and shown in shuffled order [docs/p3_gradcam_review.md, section 1].

Session date, attendees, the guide's comments and agreed actions: TBD: waiting on the signed minutes in
docs/p3_gradcam_review.md. Summary of findings (attention on lesions vs. outside the lungs, patterns on errors,
any sign of shortcut learning): TBD: waiting on docs/p3_gradcam_review.md sections 3–5.
