# D08 (CLAHE) and D09 (crop margin): the full sweep, and what should decide each

**Owner:** Person 2 · **Date:** 2026-10-14 · **Decides:** D08, D09 (provisional — rerun on
P1's C01 before locking) · **Evidence:** `artifacts/p2/dryrun/d08_d09_sweep.json`,
`scripts/p2_08_09_sweep_variants.py`

## The sweep

All ten variants, identical head, seeds (17, 29, 43, 61, 79), early stopping on development,
paired bootstrap with 2,000 resamples against the plain full image. 8,400 images;
development 659, of which 59 TB.

| variant | dev AUC | sens @ 95% spec | ΔAUC vs full image | 95% CI | verdict |
|---|---|---|---|---|---|
| **full_clahe0** | **0.9949** | **0.9661** | — | — | reference |
| crop_m0.15_clahe0 | 0.9944 | 0.9661 | −0.0006 | [−0.0019, +0.0002] | no difference |
| crop_m0.10_clahe0 | 0.9941 | 0.9661 | −0.0009 | [−0.0029, +0.0003] | no difference |
| crop_m0.05_clahe0 | 0.9939 | 0.9661 | −0.0012 | [−0.0030, +0.0000] | no difference |
| crop_m0.00_clahe0 | 0.9931 | 0.9661 | −0.0019 | [−0.0046, +0.0000] | no difference |
| full_clahe1 | 0.9920 | 0.9492 | −0.0030 | [−0.0079, +0.0001] | no difference |
| crop_m0.15_clahe1 | 0.9907 | 0.9492 | −0.0042 | [−0.0111, −0.0002] | **worse** |
| crop_m0.00_clahe1 | 0.9901 | 0.9492 | −0.0049 | [−0.0103, −0.0010] | **worse** |
| crop_m0.10_clahe1 | 0.9893 | 0.9492 | −0.0056 | [−0.0147, −0.0005] | **worse** |
| crop_m0.05_clahe1 | 0.9888 | 0.9492 | −0.0059 | [−0.0138, −0.0010] | **worse** |

Sensitivity at 70% specificity is **1.0000 for all ten variants**, as
`p2_selection_criterion_issue.md` predicted. It is reported, never selected on.

**No variant is significantly better than the plain full image.**

## D08 — CLAHE: off

The evidence is consistent rather than individually strong, and both halves of that matter:

- Every `clahe0` variant beats its `clahe1` twin on AUC, in all five pairings.
- Sensitivity at 95% specificity is 0.9661 for all five `clahe0` variants and 0.9492 for all
  five `clahe1` variants.
- Four of the five CLAHE variants are *significantly* worse than the full image on the
  paired bootstrap; the fifth (`full_clahe1`) is indistinguishable.
- **CLAHE is never better, in any variant, on any metric measured.**

Being precise about the size of this: 0.9661 versus 0.9492 is **57 of 59 TB cases versus
56 of 59 — a single case**, and the Wilson intervals overlap heavily ([0.8846, 0.9907] and
[0.8608, 0.9826]). On its own that difference decides nothing. What makes the direction
trustworthy is that it is the same in all five independent pairings and agrees with the AUC
bootstrap, where four of five are significant. "CLAHE does not help and probably hurts a
little" is the supportable claim; "CLAHE costs one TB case in sixty" is not.

**D08: CLAHE off.** Decided by the primary metric, in agreement with the tie-break — the
criterion did its job here.

## D09 — crop margin: drop cropping from the classification path

Here the amended criterion **still does not discriminate**. All five `clahe0` variants score
*exactly* 0.9661 at 95% specificity, so the AUC tie-break decides, and it does so on a
difference of 0.0006 between the best and second-best — far inside the bootstrap interval,
which spans zero.

Reporting "full image selected, by AUC tie-break" would be technically true and
substantively misleading: it implies a measurement that did not happen. The honest statement
of what the sweep found is:

> Crop margin makes no measurable difference to classification accuracy. Five margins,
> including no crop at all, are tied on the primary metric, and no crop variant is
> significantly better than the uncropped image.

Given a genuine tie, the decision should be made on a ground the data *does* support, and
simplicity is the obvious one. The full image:

- is at least as accurate as every crop variant (highest AUC, tied on the primary metric);
- removes a PSPNet segmentation forward pass per image from the classification path;
- removes a sanity-check failure mode (what to do when segmentation fails) and the fallback
  rule that goes with it;
- and removes a decision from the critical path, since the margin stops mattering.

**D09: use the full image for classification; no crop.** Decided by tie plus simplicity,
stated as such.

This does **not** remove lung segmentation from the project. It stays for Person 3's
detector and for interpretability — a lesion box or heatmap is easier to read on a lung crop
— and whether cropping helps the *detector* is a separate, untested question that this sweep
says nothing about. See `p2_does_lung_cropping_help.md`.

## Amendment to the selection criterion, part two

`p2_selection_criterion_issue.md` recommended selecting at 95% specificity instead of 70%,
because at 70% every variant scored 1.0000. With all ten variants measured, that amendment
is half-sufficient:

| decision | does sens@95 discriminate? |
|---|---|
| D08 (CLAHE on/off) | **yes** — 0.9661 vs 0.9492, consistent across all five pairings |
| D09 (crop margin) | **no** — five-way exact tie |

So the criterion should be stated with its fallback made explicit rather than left to the
tie-break:

> Select on sensitivity at 95% specificity. Where variants tie on that metric and the paired
> bootstrap shows no significant AUC difference between them, prefer the **simpler**
> pipeline, and record that simplicity — not accuracy — was the deciding ground.

That is a pre-registrable rule, it is honest about what the numbers can and cannot settle,
and it would have produced the same answer here without dressing a tie up as a measurement.

## Status

Provisional. The split is not Person 1's C01, so **D08 and D09 are not locked**. The sweep
takes 5.8 minutes to rerun end to end (`py -3 scripts/p2_08_09_sweep_variants.py`) once C01
lands, and that rerun is what gets locked. The earlier version of this sweep was run inline
and was not reproducible, which is not an acceptable basis for a locked decision; it is a
script now.
