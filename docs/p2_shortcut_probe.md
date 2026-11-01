# The classifier predicts TB from the border of the image

**Owner:** Person 2 · **Date:** 2026-10-16 · **Affects:** every reported number, §6, §12,
§17, and what the paper can claim · **Status:** provisional split, development; four
independent maskings

## The result

`p2_prelim_result.md` flagged the 0.99 AUC as a shortcut-learning risk and nothing tested
it. This is the test. Identical head, identical seeds, identical training procedure — only
the pixels differ. Masked regions are filled with the per-image mean, so the removed area
does not announce itself as unusually bright or dark.

| variant | what the model can see | dev AUC | sens @ 95% spec | above-chance AUC retained | ΔAUC vs full, 95% CI |
|---|---|---|---|---|---|
| **full** | everything | **0.9949** | 0.9661 | 100% | — |
| lungs_only | lung field only | 0.9889 | 0.9492 | 98.8% | −0.0060 [−0.0141, −0.0002] |
| lungs_erased | everything **except** the lungs | 0.9532 | 0.7797 | **91.6%** | −0.0423 [−0.0833, −0.0134] |
| lungs_erased_dilated | lungs grown by 48 px erased | 0.9493 | 0.7797 | **90.8%** | −0.0462 [−0.0732, −0.0229] |
| **border_only** | **outer 12% frame only** | **0.9094** | 0.5593 | **82.7%** | −0.0869 [−0.1213, −0.0566] |

**The outer 12% frame of a chest X-ray predicts TB at AUC 0.9094.** That frame contains
shoulders, upper arms, the image edge, collimation boundaries, background and any burned-in
markers. It contains no lung tissue under any plausible segmentation error.

## Why this is not an artifact of imperfect masking

The obvious objection to `lungs_erased` is that lung segmenters trim the **apices**, and TB
is classically an upper-lobe disease — so the "lung-free" image might still contain real
pathology. Two controls rule that out:

1. **Dilating the erased region by 48 px (~10% of the image) costs only 0.0039 AUC**
   (0.9532 → 0.9493). If apical leakage were carrying the signal, generously removing the
   apices and the peri-hilar margin would have broken it. It did not.
2. **`border_only` reaches 0.9094 with no anatomy of the lungs at all.** This is the
   decisive variant: there is no segmentation-error story that puts lung pathology in the
   outer frame of the image.

A third, independent line of evidence points the same way. In the robustness sweep
(`p2_robustness.md`), **horizontally flipping the image changes nothing** (ΔAUC +0.0001
[−0.0005, +0.0010]). A chest radiograph is strongly asymmetric — heart left of midline,
three lobes on the right against two on the left, lateralised aortic arch and gastric
bubble. A model reading anatomy would notice a mirror. This one is indifferent.

And a fourth: **`lungs_only` is significantly *worse* than the full image** (0.9889 vs
0.9949, CI excluding zero). If the task were being solved from lung pathology, handing the
model clean isolated lungs should not hurt it.

## What this means

**The honest statement: a large majority of this classifier's discriminative performance
does not come from lung pathology.** 82.7% of its above-chance AUC is available from the
image border alone.

What the border plausibly encodes: collimation and field-of-view settings, patient
positioning and body habitus, exposure and post-processing, device markers — in short, *how
and where the image was taken*, and therefore **which group of patients it was taken from**.
TBX11K's TB cases and its healthy/sick controls were collected as distinct groups. Anything
that differs systematically between those acquisition groups is available to the model as a
free signal, and it is using it.

This is the same failure mode as a livestock-breed classifier scoring 98% off the barn
backdrop rather than the animal. It is not a bug in the code; it is the dataset's structure
being learned faithfully.

## Consequences across the project

**§6 / §17 — the headline AUC.** 0.9949 internal and the near-parity with published
SymFormer numbers (99.1%) must now be read with this attached. It does not mean our pipeline
is broken; it raises the question of how much of *any* reported TBX11K number is the same
effect. Our §17 section already declines to claim parity, for other reasons; this is a
stronger reason.

**§4.7 — it explains the cross-site failure.** A model keyed on acquisition characteristics
is precisely a model that separates TBX11K from CheXpert at AUROC ≈ 1.0, and that degrades
at a new site. Two findings that looked independent are one finding.

**External validation — the number needs re-examining.** Shenzhen gives AUC 0.9382
(`p2_external_validation.md`), which looked like evidence of genuine generalisation. But
Shenzhen also collected its TB and normal cases as separate groups, so the same shortcut may
be available there. **The probe must be repeated on Shenzhen**: train on TBX11K borders only,
test on Shenzhen. If border-only transfers, the external number is partly non-pathological
too. This is the single most important remaining experiment and it is cheap.

**D09 — the crop decision is now better supported, for a new reason.** D09 chose the full
image over lung crops on simplicity grounds, because accuracy was tied
(`p2_d08_d09_recommendation.md`). That tie now looks different: the full image ties with the
crops *while also having access to the shortcut*. A lung crop is the variant that
structurally cannot use the border. **Recommendation: reopen D09** and consider the tight
crop (`crop_m0.00`) not because it is more accurate — it is not, −0.0019 — but because it
removes the shortcut's substrate. That is a validity argument rather than an accuracy one,
and it should be made explicitly.

## What to do

1. **Report it.** This belongs in the abstract, not buried in limitations. A project that
   finds and quantifies its own shortcut is doing science; one that reports 0.99 and stays
   quiet is not.
2. **Repeat the probe on Shenzhen** before citing the external number as evidence of
   pathological generalisation.
3. **Reconsider D09 in favour of the lung crop** on validity grounds, and re-measure the
   external AUC under the crop. If cropping costs 0.002 internally and recovers external
   validity, that is an easy trade.
4. **Do not pseudo-label** anything with this model. Whatever it is keying on would be baked
   into the labels.

## Caveats

- Development split, provisional C01, 59 TB cases — the sensitivity column is coarse.
- Masking with the per-image mean leaves a visible mask *shape*, and the lung outline is
  itself weakly informative (lung size and position correlate with body habitus and disease).
  `border_only` is immune to this objection, since its kept region is a fixed rectangle
  identical for every image — which is part of why it is the variant to trust.
- This quantifies that non-lung signal *suffices*; it does not prove the full model ignores
  the lungs. `lungs_only` at 0.9889 shows lung tissue is also sufficient on its own. Both
  routes work, and the model is free to use either.

Evidence: `artifacts/p2/shortcut/shortcut_probe.json`, reproduced by
`scripts/p2_19_shortcut_probe.py`.
