# Robustness: the AUC is stable and the operating point is not

**Owner:** Person 2 · **Date:** 2026-10-16 · **Covers:** the §12 limitations and §20 hardware
narrative · **Status:** provisional split, development only

The head is trained once on clean training data and frozen; only the input varies, as in
deployment. The threshold is likewise fixed at 95% specificity on clean development, because
a deployed system does not get to re-tune per image. 16 input conditions, 659 development
images, paired bootstrap against clean.

## 1. AUC barely moves

| condition | AUC | ΔAUC | 95% CI | verdict |
|---|---|---|---|---|
| clean | 0.9950 | — | — | — |
| rotation ±5°, ±10° | 0.9933–0.9953 | −0.0017 to +0.0003 | all cross zero | no difference |
| JPEG q90 / q70 / q50 | 0.9948 / 0.9945 / 0.9937 | −0.0002 to −0.0013 | all cross zero | no difference |
| gamma 0.8 / 1.25 | 0.9943 / 0.9945 | −0.0007 / −0.0005 | cross zero | no difference |
| blur σ1.0 / σ2.0 | 0.9949 / 0.9948 | −0.0001 / −0.0002 | cross zero | no difference |
| horizontal flip | 0.9951 | +0.0001 | [−0.0005, +0.0010] | no difference |
| noise σ5 | 0.9924 | −0.0026 | [−0.0064, −0.0000] | **worse** |
| noise σ10 | 0.9854 | −0.0096 | [−0.0228, −0.0008] | **worse** |
| noise σ20 | 0.9663 | −0.0287 | [−0.0610, −0.0071] | **worse** |

Only additive noise degrades ranking performance, and even at σ = 20 the AUC is 0.9663. On
this evidence alone the model looks robust, and that is what a paper would normally report.

## 2. The operating point collapses, and AUC cannot see it

The same runs, now looking at specificity at the **fixed deployed threshold**:

| condition | sensitivity | **specificity** |
|---|---|---|
| clean | 0.9661 | **0.9500** |
| noise σ5 | 0.9661 | 0.8950 |
| noise σ10 | 0.9661 | **0.6767** |
| noise σ20 | 0.9831 | **0.2583** |

Under noise σ = 20, **specificity falls from 95% to 25.8%** while AUC only drops 0.029 and
sensitivity actually *rises*. The score distribution shifts bodily upward: noise pushes
everything toward "TB". In a clinic, that is a system which starts flagging roughly
three-quarters of healthy people as positive, while its AUC still reads 0.97.

**This is the finding worth reporting.** AUC is invariant to any monotone transformation of
the scores, so a uniform upward shift is exactly the failure it is blind to. A statement like
"robust to noise, AUC 0.966" would be true and seriously misleading.

Two consequences:

1. **Any robustness claim must be made at a fixed threshold, not in AUC.** The report should
   give both columns above, and §12 should state the specificity collapse explicitly.
2. **A deployed gate should screen for noise.** The quality gate (D07) currently checks
   blankness, exposure and aspect ratio — not noise. A noise estimator (for instance the
   median absolute Laplacian response) would catch exactly the condition that breaks the
   operating point, and it is the one failure mode here that the gate could prevent but does
   not. Recommended addition to D07, with the caveat that it adds a fifth C09 reason and so
   needs a contract change agreed with Person 4.

## 3. The flip result, and what it says about what the model reads

**Horizontal flipping changes nothing: ΔAUC +0.0001 [−0.0005, +0.0010].**

A chest radiograph is not symmetric. The heart sits left of midline, the right lung has three
lobes to the left's two, the aortic arch and gastric bubble are lateralised. A model reading
anatomy should notice a mirror; a radiologist certainly would. This one is entirely
indifferent.

That is consistent with the shortcut probe (`p2_shortcut_probe.md`), which finds TB
predictable at AUC 0.9532 from images with the lung field erased. Two independent experiments
pointing the same way: whatever the classifier is keying on is not lateralised anatomical
structure.

It also has a practical upside worth stating plainly: **the system is immune to the
left-right flip that is a common real data-entry error.** That is a genuine robustness
result, even though its cause is not reassuring.

## 4. What this does not cover

- One site, one scanner population (TBX11K), development split only.
- Synthetic perturbations. Real scanner noise is not additive Gaussian, real motion blur is
  directional, and real rotation comes with cropping and different collimation.
- No perturbation combinations; real degradation arrives in combinations.
- 59 TB cases, so sensitivity moves in steps of 1/59 and is too coarse to show small effects
  — which is why it looks pinned at 0.9661 across almost every condition.
- The noise-induced shift was measured at one threshold. How it interacts with Person 4's
  calibration (D22) is untested, and recalibration might absorb part of it.

Evidence: `artifacts/p2/dryrun/robustness.json`, reproduced by
`scripts/p2_21_robustness.py`.
