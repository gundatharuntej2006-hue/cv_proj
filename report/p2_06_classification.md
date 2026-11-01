# §6 Classification

*Owner: Person 2. Every number cites its artifact. All results are on the **provisional**
split (Person 1's C01 pending) and the **development** split; `internal_test` has not been
read — see §12 and `docs/p2_sealed_scoring.md`.*

## 6.1 Architecture

A **frozen** RAD-DINO backbone with a **linear head** on the pooled embedding.

- Backbone: `microsoft/rad-dino`, revision `110cbc18d5133582e320b43d53bf5c44e410c936`,
  MIT licence, ViT-B/14 at 518 px → 1,370 tokens, 768-dimensional pooled output (D15).
- Head: `Linear(768 → 2)`, trained on L2-normalised embeddings.
- Nothing in the backbone is trained. Embeddings are extracted once per preprocessing
  variant and cached, so a head trains in seconds and five seeds cost minutes.

Freezing was not a compromise forced by the 6 GB laptop — it was verified as sufficient in
Week 1 (P2-01: embedding peak 0.67 GB at batch 16; the Grad-CAM forward-plus-backward path
peaks at 0.84 GB, against a 5.5 GB budget; `docs/p2_vram_feasibility.md`). It also makes
every downstream experiment cheap and exactly reproducible, which is why ten preprocessing
variants and five seeds each were affordable at all.

## 6.2 Class imbalance (D10)

Exactly two methods are permitted, and the training config *raises* on anything else rather
than silently accepting it (`p2_cls.heads.TrainConfig.__post_init__`): a weighted sampler,
or logit-adjusted loss (Menon et al.). The reported models use **logit-adjusted loss**.
Training length is set by early stopping on development, never by a fixed epoch count.

Five seeds (17, 29, 43, 61, 79) are averaged. The seed-to-seed spread of development AUC is
**0.0001**, so none of the differences discussed below are seed noise
(`artifacts/p2/dryrun/d08_d09_sweep.json`).

## 6.3 Head type (D11): binary, with three-class as auxiliary

Both heads were trained identically and judged **only on the TB decision**, since that is
the only output the triage consumes:

| head | AUC on the TB label | sens @ 95% spec | sens @ 70% spec | seed sd |
|---|---|---|---|---|
| **binary (2-class)** | **0.9950** | 0.9661 [0.8846, 0.9907] | 1.0000 | 0.0001 |
| three-class (healthy / sick_non_tb / tb) | 0.9921 | 0.9661 [0.8846, 0.9907] | 1.0000 | 0.0002 |

Paired bootstrap, three-class minus binary: **−0.0028, 95% CI [−0.0079, −0.0000]**.

The interval's upper bound rounds to zero, so this is at the very edge of detectability, and
the primary metric is **identical** (0.9661 for both — the same 57 of 59 TB cases). The
honest reading is that the three-class head costs nothing measurable and gains nothing
measurable on the TB decision.

**D11: binary head for the TB decision; the three-class head is retained as an auxiliary
output** for the dashboard's healthy / sick-but-not-TB / TB display, which is what it was
wanted for. Source: `artifacts/p2/dryrun/section14_ablations.json`.

## 6.4 Preprocessing (D08, D09)

Ten variants — full image plus four crop margins (0.00, 0.05, 0.10, 0.15), each with CLAHE
off and on — all with the identical head, seeds and criterion. Full table in
`docs/p2_d08_d09_recommendation.md`; summary:

| | best variant | dev AUC | sens @ 95% spec |
|---|---|---|---|
| CLAHE off | **full_clahe0** | **0.9949** | **0.9661** |
| CLAHE on | full_clahe1 | 0.9920 | 0.9492 |

**D08 — CLAHE off.** Every `clahe0` variant beats its `clahe1` twin on AUC in all five
pairings; four of five CLAHE variants are significantly worse than the full image on the
paired bootstrap; CLAHE is never better on any metric measured. The effect is small —
0.9661 versus 0.9492 is 57 of 59 TB cases against 56, with heavily overlapping Wilson
intervals — so the supportable claim is "CLAHE does not help and probably hurts slightly",
not a quantified cost per case. What makes the direction trustworthy is its repetition
across five independent pairings.

**D09 — full image, no crop.** All five `clahe0` variants score *exactly* 0.9661 at 95%
specificity, and **no variant is significantly better than the uncropped image**. The AUC
tie-break would decide on a difference of 0.0006, well inside its own bootstrap interval, so
reporting "selected by AUC" would dress a tie as a measurement. The decision is made on the
ground the data does support — simplicity. The full image is at least as accurate and removes
a PSPNet segmentation pass, a sanity-check failure mode and a fallback rule from the
classification path.

Lung segmentation **remains** in the project, for Person 3's detector and for
interpretability. Whether cropping helps the *detector* is a separate, untested question
(§7).

## 6.5 Backbone choice, measured (§14 backbone row)

"We used RAD-DINO" is not a finding. The comparison a reviewer would ask for is a DenseNet
trained on 800k chest X-rays, one `pip install` away. Both backbones were given
byte-identical pixels (the same `load_manifest`/`read_image` path), the identical head, the
same seeds and the same criterion:

| model | dim | dev AUC | sens @ 70% | sens @ 95% | ΔAUC vs RAD-DINO | 95% CI |
|---|---|---|---|---|---|---|
| **RAD-DINO + linear head** | 768 | **0.9949** | 1.0000 | **0.9661** | — | — |
| XRV DenseNet-121 + linear head | 1024 | 0.9539 | 0.9661 | 0.8475 | −0.0409 | [−0.0727, −0.0140] |
| XRV 18-d pathology vector + head | 18 | 0.9024 | 0.9153 | 0.5932 | −0.0897 | [−0.1338, −0.0515] |
| XRV zero-shot, best proxy pathology | — | 0.8026 | — | — | −0.1924 | [−0.2541, −0.1328] |

**RAD-DINO wins and every interval excludes zero.** Two details belong in the report:

- The DenseNet has **no tuberculosis label**. The best zero-shot proxy is **Pneumonia at AUC
  0.8026**, then Nodule (0.7622). So an off-the-shelf model with no TB training reaches 0.80
  — the floor any claimed contribution must clear to be interesting.
- The **18-dimensional pathology vector reaches AUC 0.9024**. Eighteen interpretable numbers
  recover most of the separability, which is a useful honesty check on the 768-d embedding:
  this task is mostly solvable without a large representation. The gap appears where it
  matters, at high specificity (0.5932 versus 0.9661).

Source: `docs/p2_backbone_baseline.md`, `artifacts/p2/baseline/xrv_comparison.json`.

## 6.6 Reproducibility of the reported scores (R2, D14)

Each registered model records its weight SHA256, variant, backbone revision and head type,
and the runtime API reproduces its exported scores. Measured, that requirement turned out to
be two claims with different answers:

| claim | worst absolute difference | meets 1e-4 |
|---|---|---|
| head re-run on the same stored embedding vs the exported score | **1.9 × 10⁻⁷** | yes |
| image → probability, export path vs runtime path | **5.7 × 10⁻⁴** | **no** |

The second cannot meet 1e-4 **by construction**: C07 was produced by a float16 GPU forward
pass and stored as float16, while the runtime path is float32 on CPU, and float16 carries
about three decimal digits. 5 of 8,400 cases (0.06%) sit close enough to a threshold for the
difference to change their category.

The contract is therefore stated as two numbers — 1e-4 for head reproducibility, 1e-3 for
end-to-end — and the final chosen variant will be exported in float32 so that both paths
agree exactly. Source: `docs/p2_r2_reproducibility.md`,
`artifacts/p2/dryrun/r2_verification.json`.

## 12.1 Narrative: how good is the classifier, really

On held-out data from the same source, very good: AUC 0.9920 on a sealed rehearsal run of
901 test images, against 0.9949 on development, with sensitivity 0.9703 at 95% specificity
(`docs/p2_sealed_scoring.md`). The gap between development and test is small, so the head is
not overfitted.

Three caveats matter more than that number, and the report should give them equal weight.

**It is a single-source result.** Every image comes from TBX11K. §4.7 shows how
comprehensively a model can learn a dataset's scanner signature: both backbones tested
separate TBX11K from CheXpert at AUROC ≈ 1.0. Nothing here establishes that the classifier
would hold up at a new hospital, and the honest expectation is that it would degrade.

**The task as measured is easier than the task as deployed.** Sensitivity at 70%
specificity is 1.0000 for all ten preprocessing variants, for both head types, and for
RAD-DINO — the metric saturates. Saturation is a property of the benchmark, not evidence of
clinical readiness; an 18-dimensional interpretable feature vector already reaches AUC
0.9024 on it.

**The three-way triage currently has only two ways.** The REFER band is empty — 0 of 901
cases in the sealed rehearsal — because the calibrated probabilities are strongly bimodal
(TB median 0.9997, non-TB 0.0070), so precision at the lower threshold already exceeds the
0.80 target and the upper threshold collapses onto it. A three-way triage that never uses
its middle category is a two-way triage, and that is a design question for Person 4's D23
rather than something to be worked around here (§8, §9).

What can be said without qualification: a frozen, MIT-licensed, publicly available backbone
plus a linear head trained in minutes on a laptop reaches 0.99 AUC on this benchmark and
beats a 800k-image supervised CXR CNN by a measurable margin. That is a genuine result about
how far transfer from a radiology foundation model goes. It is not a claim about screening
patients.
