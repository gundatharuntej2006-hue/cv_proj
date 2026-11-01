# §17 Research comparison, and §14 comparative-experiment summary

*Owner: Person 2. All of this project's numbers are on the **provisional** split (Person 1's
C01 pending) and are not directly comparable to published numbers on the official TBX11K
test split — see §17.2, which is the most important part of this section.*

## §14 Comparative experiments: the rows Person 2 owns

Every row uses the identical head, the same five seeds (17, 29, 43, 61, 79), the same early
stopping and a paired bootstrap of 2,000 resamples on the development split (659 images,
59 TB). Full tables in the cited artifacts.

| row | compared | result | decided | source |
|---|---|---|---|---|
| **Backbone** | RAD-DINO vs TorchXRayVision DenseNet-121 vs 18-d pathology vector vs zero-shot | 0.9949 / 0.9539 / 0.9024 / 0.8026 AUC; ΔAUC −0.0409 [−0.0727, −0.0140] vs DenseNet | **RAD-DINO**, by measurement | `p2_backbone_baseline.md` |
| **Head** | binary vs three-class | primary metric identical (0.9661 both); ΔAUC −0.0028 [−0.0079, −0.0000] | **binary**; three-class kept as auxiliary | `section14_ablations.json` |
| **CLAHE (D08)** | off vs on, across all five crop settings | `clahe0` wins every pairing; 4 of 5 CLAHE variants significantly worse | **off** | `p2_d08_d09_recommendation.md` |
| **Crop margin (D09)** | full image vs margins 0.00 / 0.05 / 0.10 / 0.15 | five-way **exact tie** on the primary metric; none significantly better than full | **full image**, on simplicity | `p2_d08_d09_recommendation.md` |
| **kNN *k* (D13)** | *k* = 1, 3, 5, 10, 20 | alien AUROC **1.0000 at every *k***, spread 0.0000 | *k* = 5 kept; **ablation found no effect** | `section14_ablations.json` |

Three of these five rows came back as "no measurable difference". That is the most useful
thing about the table: it tells the reader which parts of the pipeline are load-bearing (the
backbone, and CLAHE weakly) and which are decoration (crop margin, *k*, head type). A
comparison table in which every row shows a win is usually a table that was not really run.

## §17.1 Published results on TBX11K

The benchmark's own paper and its TPAMI extension define the state of the art. SymFormer, a
purpose-built architecture with a symmetric-attention mechanism designed for the bilateral
structure of chest X-rays, trained jointly for classification and detection:

| method | backbone | accuracy | AUC (TB) | sensitivity | specificity |
|---|---|---|---|---|---|
| SymFormer w/ RetinaNet | P2T-Small w/ FPN | 94.6% | **99.1%** | 92.1% | 96.7% |
| SymFormer w/ RetinaNet | ResNet-50 w/ FPN | 94.5% | 98.9% | 91.0% | 96.8% |
| SymFormer w/ Deformable DETR | ResNet-50 w/ FPN | 94.3% | 98.5% | 87.3% | 97.3% |

For a lower reference point, a 2026 audit of medical vision-language models reports
zero-shot AUROC on TBX11K validation of 0.800 (MedSigLIP), 0.795 (CheXficient), 0.742
(BioMedCLIP) and 0.564 (OpenCLIP) — i.e. general-purpose multimodal models, without
TB-specific training, sit near 0.80 at best.

**This project**, frozen RAD-DINO plus a linear head, sealed rehearsal on 901 held-out
images of the provisional split: **AUC 0.9920**, sensitivity 0.9208 at specificity 0.9888 at
the fitted operating point, sensitivity 0.9703 at 95% specificity
(`docs/p2_sealed_scoring.md`).

## §17.2 Why that is not a like-for-like comparison, and must not be reported as one

The numbers above sit within a whisker of each other — 0.9920 against 0.9910 — and it would
be easy to write "matches the state of the art on a laptop". **That claim is not supported**,
for four reasons, every one of which should appear next to any such table.

**1. Different test data.** SymFormer's numbers are on the official TBX11K test split. This
project's are on a re-split of trainval, made because the official test labels were not
available to us. The two are different image sets, so the comparison is between *methods on
different data*, not between methods.

**2. The official splits contain cross-split duplicates.** We found **158 near-duplicate
pairs at cosine ≥ 0.95 spanning the official split boundaries, 52 of them train→val**
(`docs/p2_duplicate_finding.md`). Any published number that trained on the official train
split and evaluated on the official val split is therefore inflated to an unknown degree.
This is a finding about the benchmark, not an accusation about any paper — the duplicates
are in the distributed data — but it means the published figures are an unreliable yardstick,
and so is any comparison drawn against them.

**3. SymFormer is solving a harder problem.** It localises lesions as well as classifying,
and its classification head is one output of a multi-task model. A classification-only
comparison flatters the single-task method. Our project's detection work (§7) is a separate
component and is not what produced the 0.9920.

**4. The benchmark's classification task is close to saturated.** Sensitivity at 70%
specificity is 1.0000 here for **all ten** preprocessing variants, both head types and the
best DenseNet configuration; an 18-dimensional interpretable feature vector reaches AUC
0.9024. When a metric cannot distinguish ten variants of one method, it cannot meaningfully
rank two different methods either. The differences between the published entries in §17.1
(98.5% to 99.1% AUC) are of the same order as the differences we measure *between
preprocessing choices of a single model* (0.9888 to 0.9949) — which is to say, not clearly
differences at all.

## §17.3 What this project can honestly claim

Stated carefully, there are three results worth reporting:

**A frozen, MIT-licensed, publicly available radiology foundation model plus a linear probe
is competitive with purpose-built architectures on this benchmark.** No fine-tuning, five
seeds, minutes of training on a 6 GB laptop GPU. Whether that is a statement about RAD-DINO
or about TBX11K is genuinely open — and §17.2 suggests it is partly about TBX11K.

**The backbone choice is the only pipeline decision that measurably matters.** RAD-DINO beats
a DenseNet-121 trained on 800k chest X-rays by ΔAUC 0.0409 with an interval excluding zero,
while crop margin, kNN *k* and head type make no measurable difference. For anyone building
a similar system, that is the actionable finding: spend the effort on the representation, not
on the preprocessing.

**The benchmark has two problems that limit what any result on it means.** It contains
cross-split duplicates, and both backbones we tested separate it from another hospital's data
at AUROC ≈ 1.0 (§4.7) — so a model trained on it learns a scanner signature along with the
pathology. The second point is not specific to this project or to RAD-DINO: mean-removal,
whitening, lung cropping and an entirely different backbone all left it untouched.

## §17.4 For a future paper

The negative results are more publishable than the positive one, because the positive one is
a benchmark number and the negative ones are about method validity:

1. **A kNN OOD gate calibrated on one public dataset does not transfer.** Quantified:
   98.7% false-rejection at a new site, AUROC 0.9837 for the score itself, ~25 local images
   to recover it conformally, and four attempted corrections that all fail. Papers reporting
   such a gate on a single source should be expected to show a second site.
2. **A conformal threshold used as a target rather than a bound discards valid data for no
   benefit.** Here it threw away 9.67% of sick and 3.39% of TB cases while alien detection
   was saturated at 100% well above it (§4.4). The error is easy to make and invisible in
   every accuracy metric, because rejected cases carry no probability and are therefore
   absent from sensitivity.
3. **TBX11K's cross-split duplicates** should be documented for the community, with the
   candidate list (`artifacts/p2/dedup/provisional_near_dup_official_v0.csv`, τ locked
   before measurement per D06).

Sources:
[Revisiting Computer-Aided Tuberculosis Diagnosis (IEEE TPAMI)](https://arxiv.org/abs/2307.02848) ·
[yun-liu/Tuberculosis](https://github.com/yun-liu/Tuberculosis) ·
[Auditing Medical Vision-Language Models for CXR TB Screening](https://arxiv.org/html/2609.21763)
