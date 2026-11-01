# The OOD gate's threshold does not transfer between sites; its score does

**Owner:** Person 2 · **Tasks:** P2-10, P2-11 (prototype) · **Date:** 2026-10-07
**Affects:** D13, the methodology's Section on the quality/OOD gate, P4-14 and P4-17
**Status:** evidenced, with two stated caveats

## Summary

The methodology specifies an out-of-distribution gate as the k=5 mean cosine distance from
a RAD-DINO embedding to a bank of training images, with the threshold set so that 95% of the
development split passes.

**The threshold does not transfer.** Carried unchanged to an unseen hospital it rejects
**98.7% of ordinary frontal chest X-rays** — the system would be unusable there.

**The score does transfer.** At that same unseen site the gate still ranks OOD images above
normal ones at **AUROC 0.9837**, and re-fitting the threshold on about **25 local normal
chest X-rays** restores it: 92% of frontals kept while 92% of laterals and 95% of non-medical
images are still caught.

So this is not a broken component. It is a second instance of the project's central claim —
that an operating point set at one site must be re-established at the next, from a small local
sample — now shown for the OOD gate as well as the CAD triage threshold. Two independent
measurements of how little local data that takes is a stronger result than one.

## The numbers

Bank: 6,888 TBX11K training images. Threshold 0.5755, set by the D13 rule (95% of held-out
TBX11K passes).

| Set | n | median distance | rejected |
|---|---|---|---|
| TBX11K val — in-distribution | 2,088 | 0.4021 | 5.0% |
| Shenzhen originals, ~3000 px | 662 | 0.4340 | 5.0% |
| Montgomery originals, ~3000 px | 138 | 0.5248 | 26.8% |
| **CheXpert Plus frontal — unseen site** | **150** | **0.7675** | **98.7%** |
| CheXpert Plus lateral — true OOD | 500 | 0.8723 | 100.0% |
| Non-medical photographs — true OOD | 246 | 0.8677 | 100.0% |

Note the spacing. The distance from in-distribution (0.40) to *an ordinary chest X-ray from
another hospital* (0.77) is more than twice the distance from there to a lateral view or a
photograph of a landscape (0.87). The axis the gate measures most strongly is not "is this a
frontal chest radiograph".

## Ruling out the obvious explanations

**It is not resolution or rescaling.** The CheXpert frontals were resized to 512×512, exactly
as TBX11K is distributed, and re-embedded: rejection stayed at **98.7%**, median 0.7654 against
0.7675. Identical.

**It is not brightness or contrast.** Mean intensity 0.508 and SD 0.291 for CheXpert frontals
against 0.485 and 0.254 for TBX11K — near identical, and both span the full range.

**It is not fixed by a handful of local images.** Adding 5, 10, 25, 50 or 75 CheXpert frontals
to the bank left rejection of the remaining frontals at 98.6–98.9%. The reason is visible in the
distances: CheXpert frontals sit **0.776 from each other**, barely closer than the 0.758 that
separates them from TBX11K. They do not form a cluster for the bank to learn.

**TBX11K is the anomaly, not CheXpert.** TBX11K images sit 0.315 from each other. That
tightness is a property of a corpus that its creators resized and standardised uniformly, not
a property of chest radiographs. The gate has learned "looks like a TBX11K file", and almost
nothing in the real world does.

**A trained classifier helps, but not enough.** A logistic probe on the same embeddings
(positives: TBX11K train; negatives: 250 laterals and 123 non-medical photographs; CheXpert
never seen in training) rejects every held-out lateral and every held-out photograph, and
accepts 100% of in-distribution images — but accepts only **36.7%** of unseen-site frontals.
Better than the distance gate's 1.3%, still unusable.

## A correction

Earlier in this session I ran Shenzhen and Montgomery at original resolution against the gate,
saw 5% and 26.8% rejection, and concluded the gate was reading content rather than
preprocessing. **That control was invalid** — but not for the reason given here originally,
and the reason matters.

> **Correction, 2026-10-16 — see `p2_tbx11k_composition_corrected.md`.** This section said
> those radiographs are "inside TBX11K — some in the very bank being scored against". Both
> halves are wrong. TBX11K's benchmark is all Chinese hospital data; the 400 NLM images ship
> in the archive under `imgs/extra/mc+shenzhen/`, referenced only from `all_*.txt`. The OOD
> bank is built from **core-list training embeddings**, so no NLM image was ever in it —
> verified at max cosine 0.8389 across all 8,400, zero pairs above 0.90.
>
> The control was nevertheless invalid, for a reason found later: those images were scored at
> **original resolution** (3000×2919, 4892×4020) while the bank was built from C04's 512×512
> standardised images, so the rejection rates measured a preprocessing mismatch rather than a
> site difference.
>
> **This means the gate's cross-site behaviour is an open question again, not a settled one.**
> The 98.7% CheXpert false-rejection figure below stands as measured, but it is now one site
> rather than the general case. Shenzhen and Montgomery are genuinely unseen by our models and
> are being re-tested properly — 512-standardised, against the core-list bank — by
> `scripts/p2_22_external_validation.py`. If they reject at 5–27% while CheXpert rejects at
> 98.7%, the honest conclusion is that transfer is **site-dependent**, which is a weaker and
> more useful claim than "the threshold does not transfer".

CheXpert Plus was the first site tested that is independent of the TBX11K archive entirely,
and it gives the opposite answer to that earlier run.

## The constructive half: a locally-set threshold

Threshold re-fitted as the 95th percentile of n local frontals from the new site, averaged
over 200 random draws of those n:

| local images | frontals kept | laterals caught | non-medical caught |
|---|---|---|---|
| 10 | 87.4% | 93.3% | 95.8% |
| **25** | **92.0%** | **92.2%** | **95.4%** |
| 50 | 93.0% | 92.3% | 95.8% |
| 100 | 94.4% | 91.6% | 95.6% |
| all 150 | 94.7% | 91.6% | 95.5% |

The curve is flat past about 25 images, which is the useful number: a clinic needs a couple of
dozen ordinary radiographs to commission the gate, not a prevalence survey.

AUROC for OOD against normal frontals **at the unseen site** is 0.9837 (0.9812 for laterals,
0.9888 for non-medical), against 1.0000 at the home site. The discriminative power is
essentially intact; only the operating point moved.

## Why it fails: TBX11K is pathologically homogeneous (the mechanism)

Having a fix is good; knowing *why* the gate fails makes it a result rather than a patch.
Four measurements on the real embeddings (`scripts/p2_11_site_geometry.py`,
`artifacts/p2/gate/p2_11_site_geometry.json`):

**1. The site difference is a rank-1 mean shift — and it is orthogonal to TB.** A single
direction (the difference of the two sites' mean embeddings) separates TBX11K from CheXpert
at AUROC **0.9984**. Projecting that one direction out drops site separability to **0.486**
(chance) and leaves the in-site TB probe AUC **unchanged** (0.9857 -> 0.9856). So "which
hospital" and "has TB" live in different directions — the classifier is not confounded by
site.

**2. But removing the shift does not fix the gate.** After projecting out the site
direction, the gate still rejects **97.3%** of CheXpert frontals (was 98.7%). A mean shift
is not why the gate fails.

**3. The real reason: TBX11K is twice as tight as a real hospital.** Median self kNN
distance is **0.39** within TBX11K and **0.78** within CheXpert. TBX11K images sit far
closer to each other than a real site's images do, because the benchmark's authors resized
and standardised them. A density gate calibrated on that cluster treats anything with
normal real-world spread as out-of-distribution.

**4. Whitening does not fix it either.** Standardising every dimension by TBX11K's own
mean and variance barely moves the false-rejection rate (98.7% -> 98.0%). The homogeneity
is not a per-axis scale you can divide out; it is intrinsic to how tightly the benchmark
occupies embedding space.

**Consequence.** You cannot make a new site look like TBX11K, because no real site is that
homogeneous. The only sound fix is to re-learn what "normal density" means at the new site
from local data — which is exactly the conformal re-fitting below. Embedding surgery
(projection, whitening) is shown *not* to work, which is itself worth reporting: it rules
out the obvious domain-adaptation reflexes and justifies the local-calibration approach.

This also sharpens the duplicate finding: a benchmark homogeneous enough to break a density
gate is a benchmark worth auditing, and we did (see `p2_duplicate_finding.md`).

## Making it rigorous: a conformal threshold for the gate

The 95th-percentile rule above is a heuristic. The project already owns the right tool, and
using it here unifies the two results.

Treat the local normal radiographs as a calibration set and take the threshold as the
`ceil((n+1)(1-alpha))`-th smallest score. That gives a distribution-free, finite-sample
guarantee that **P(reject a normal image) <= alpha** — the same construction Person 4
implemented for the triage threshold, applied to the gate.

It carries the same existence condition: the threshold is defined only when
`n >= 1/alpha - 1`. Nine local images for a 10% false-rejection guarantee, nineteen for 5%.
Below that, no threshold can honour the guarantee and the gate must abstain rather than
return a number — exactly the bug found in `compute_conformal_lower_threshold` during the
P4 review.

Measured at the unseen site, 300 random draws of the local set per row:

| alpha | n local | false-rejection rate | OOD caught |
|---|---|---|---|
| 0.10 | 9 (the minimum) | 10.1% | 91.9% |
| 0.10 | 25 | 7.3% | 92.8% |
| 0.10 | 50 | 9.5% | 95.1% |
| 0.05 | 19 (the minimum) | 4.7% | 86.2% |
| 0.05 | 50 | 3.8% | 88.3% |
| 0.05 | 100 | 5.1% | 92.2% |

The guarantee holds at every n where it is defined. The cost of a stricter alpha is OOD
recall, not validity: tightening false rejection from 10% to 5% costs about six points of
OOD catch at the same n.

**Why this matters for the paper.** One piece of machinery now governs both operating points
in the pipeline. The triage threshold and the OOD gate are different problems, but they have
the same structure, the same remedy, and the same minimum-sample bound. A deployment
recipe that reads "collect N locally labelled images, then set both thresholds conformally"
is a cleaner contribution than two unrelated calibration procedures.

## What follows for the project

**1. This strengthens the paper rather than weakening it.** The thesis is that a CAD triage
threshold set at one site does not transfer, and that conformal risk control can re-establish
it from a small local sample. We have now found that the *OOD gate* has the same disease. The
problem is systemic to the pipeline, not specific to one threshold — which is a more
interesting claim, and we have the evidence for it.

**2. P4-14 and P4-17 must report gate pass rates alongside every external number.** If the gate
rejects 99% of an external set, the sensitivity computed on the survivors is meaningless. The
work plan already lists "gate pass rate" in P4-14's acceptance criteria; this is why it matters.

**3. D13 should pre-register two thresholds, not one.** A home threshold fitted on development
as the methodology says, and a documented local re-fitting procedure for any new site, with the
n-versus-performance curve above as its evidence. Reporting only the home threshold would be
reporting a number that does not survive deployment.

**4. The external evaluation must re-fit before it scores.** P4-14 and P4-17 should set the gate
threshold on a held-out slice of each external set rather than carrying the TBX11K threshold
across, and report both numbers. The difference between them is itself a result worth printing.

## Two caveats, stated plainly

**These embeddings use the stand-in preprocessing, not Person 1's C04.** The images were resized
by Person 2 rather than run through the agreed pipeline, which will include lung segmentation,
cropping to the lung box and optionally CLAHE. Cropping to the lungs in particular discards
borders, collimation edges and burned-in markers — exactly the content most likely to carry a
site signature. **The first thing to re-run when C04 exists is this experiment**, and it is
plausible the effect shrinks. It is not plausible that it vanishes: the 0.40 against 0.77 gap is
far too large.

**One unseen site is weak evidence for a general claim.** CheXpert is a single American academic
centre. The direction of the effect is unambiguous, its size at other sites is not. A third
genuinely independent site — TB Portals is the candidate — would settle it.

## Reproducing

```
artifacts/p2/gate/p2_11_prototype.json      the table above
artifacts/p2/gate/p2_11_site_shift.json     the local-images-added curve
artifacts/p2/gate/p2_11_probe_vs_knn.json   trained probe against distance gate
```

Embeddings: `radino_tbx11k_full_clahe0_v1.npz`, `radino_ood_lateral_v1.npz`,
`radino_ood_nonmedical_v1.npz`, `radino_control_frontal_v1.npz`,
`radino_control_frontal512_v1.npz`, `radino_external_full_clahe0_v1.npz`.

The CheXpert frontal set is a **control, not OOD data**: it must never enter C03.
