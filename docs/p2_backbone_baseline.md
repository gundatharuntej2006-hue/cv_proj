# Is RAD-DINO worth it? Against a published CXR CNN, yes — but it does not fix the site problem

**Owner:** Person 2 · **Date:** 2026-10-14 · **Covers:** P2-18 · **Status:** provisional
split, real embeddings for both backbones

## Why this comparison exists

The methodology picks a frozen RAD-DINO backbone. Without a baseline, "development AUC
0.9949" is unfalsifiable — a reviewer's first question is what a DenseNet trained on 800k
chest X-rays would do, because that is one `pip install` away. TorchXRayVision's
`densenet121-res224-all` is that model.

Both backbones see **byte-identical pixels**: the script calls Person 2's own
`embed.load_manifest` and `read_image`, so the comparison measures backbones and not
preprocessing. Both get the identical linear head, the same five seeds, the same early
stopping and the same selection criterion.

## 1. Accuracy

Development split, 659 images, 59 TB. Paired bootstrap, 2,000 resamples.

| model | feature dim | dev AUC | sens @ 70% spec | sens @ 95% spec | ΔAUC vs RAD-DINO | 95% CI |
|---|---|---|---|---|---|---|
| **RAD-DINO + linear head** | 768 | **0.9949** | 1.0000 | **0.9661** | — | — |
| XRV DenseNet + linear head | 1024 | 0.9539 | 0.9661 | 0.8475 | **−0.0409** | [−0.0727, −0.0140] |
| XRV pathology vector + head | 18 | 0.9024 | 0.9153 | 0.5932 | −0.0897 | [−0.1338, −0.0515] |
| XRV zero-shot, best pathology | — | 0.8026 | — | — | −0.1924 | [−0.2541, −0.1328] |

**RAD-DINO wins, and the margin is real** — every interval excludes zero. This is a
genuine justification for the backbone choice, which the project previously did not have.

Two details worth keeping in the report:

- **The zero-shot baseline is weak but not useless.** None of the DenseNet's 18 pathologies
  is tuberculosis. The best proxy is **Pneumonia (AUC 0.8026)**, then Nodule (0.7622) and
  Emphysema (0.7450). So an off-the-shelf model with no TB training and no training budget
  still gets to 0.80 — which is the number any claimed improvement has to clear to be
  interesting at all.
- **The 18-d pathology vector reaches AUC 0.9024.** Eighteen interpretable numbers recover
  most of the separability. That is a useful honesty check on the 768-d embedding: the task
  does not need a large representation to be mostly solved, which fits the shortcut-learning
  worry in `p2_prelim_result.md`. The gap at high specificity is where it matters — 0.5932
  against 0.9661 sensitivity at 95% specificity.

## 2. The cross-site failure is not a RAD-DINO artifact

`p2_ood_gate_finding.md` reported that the OOD gate's threshold does not transfer: TBX11K
images sit at a k=5 cosine distance of ~0.36 from each other while a new hospital's normal
chest X-rays sit at ~0.78, so a threshold calibrated on TBX11K rejects ~99% of the new
site. The open question was whether that is a property of **RAD-DINO's** embedding space or
of **the data**. If it were the backbone, swapping backbones would fix it.

Measured in both spaces with the same k=5 OOD score, bank of 4,000 TBX11K training images,
1,000 held-out TBX11K images against 150 CheXpert frontals:

| | TBX11K self-distance | new-site distance | ratio | site separability (AUROC) |
|---|---|---|---|---|
| RAD-DINO | 0.3582 | 0.7787 | 2.17× | **1.0000** |
| XRV DenseNet | 0.0164 | 0.0920 | 5.60× | **0.9998** |

**Both backbones separate the two sites essentially perfectly.** Switching to a conventional
supervised CXR CNN does not help; in relative terms it is *worse* (5.60× versus 2.17×).

So the conclusion stands and is now much stronger: the site signature is in the data — in
how the scanner renders the tissue — and not in the choice of representation. Together with
the earlier negative results (mean-removal, whitening and lung cropping all leave it
untouched, see `p2_does_lung_cropping_help.md`), the realistic options remain the ones
already documented: calibrate the gate's threshold on a small sample from each new site
(~25 images suffices conformally), or abstain.

**A methodological warning that comes out of this.** The raw distances above are *not*
comparable across backbones, and reading them naively inverts the conclusion. DenseNet
features are post-ReLU and therefore non-negative, so every pair of feature vectors sits in
the positive orthant, every cosine similarity is high and every distance is small **by
construction** — the DenseNet's 0.0164 self-distance does not mean its space is 20× more
homogeneous than RAD-DINO's. A first pass at this comparison used median pairwise distance
and appeared to show exactly that. Only scale-free quantities — the ratio, and the AUROC,
which is invariant to any monotone rescaling of the score — can be compared between feature
spaces with different geometric constraints.

## 3. What this means for the report

- Report the baseline table in full. It converts "we used RAD-DINO" into "we used RAD-DINO
  and measured what it buys over the obvious alternative: +0.041 AUC, CI excluding zero".
- Report the zero-shot number. A screening system that cannot beat "the Pneumonia head of
  an off-the-shelf model" is not worth deploying, and saying so makes the comparison
  credible.
- The cross-site result belongs in the limitations section as a **positive** finding about
  the problem rather than a defect of this system: no backbone tested makes a
  TBX11K-calibrated OOD threshold transfer, so per-site calibration is a requirement of the
  task, not a shortcoming of the implementation.

## Caveats

- Provisional split (Person 1's C01 pending); development only, `internal_test` untouched.
- 59 TB cases in development, so the bootstrap intervals are wide; only the sizeable effects
  here are detectable, and all three ΔAUC intervals are comfortably clear of zero.
- One external site (CheXpert) for the cross-site comparison, and only its frontal normals.
- The DenseNet is used frozen with a linear head, which matches how RAD-DINO is used. Fine
  tuning either backbone is a different experiment and is not claimed here.

Evidence: `artifacts/p2/baseline/xrv_comparison.json`, reproduced by
`scripts/p2_18_xrv_baseline.py`.
