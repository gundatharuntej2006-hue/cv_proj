# From 0.99 to 0.62: external validation is not one thing

**Draft v2 — Person 2 · 2026-10-20 · all numbers provisional pending the final split**

*v1 argued that a dataset-specific shortcut inflates internal evaluation and that the
external number is therefore the trustworthy one. The Vietnam cohort showed that
"external" conceals a ladder, and the standard external sets sit near its top. This draft
is restructured around that.*

*Target: a workshop paper (ML4H / MICCAI workshop) or arXiv preprint.*

---

## Abstract

Published methods report above 0.99 AUC on TBX11K, the largest public tuberculosis
chest-radiograph benchmark. We measure what survives as the evaluation moves from that
benchmark towards a clinic.

A frozen radiology foundation model with a linear probe reaches **0.9949** on TBX11K. It
also reaches **0.9094 from the outer 12% of the image alone** — a frame containing
shoulders, collimation and background, and no lung tissue — so 82.7% of its above-chance
performance is available without the anatomy of interest. Dilating the erased region to
remove the apices costs 0.0039, and the model is invariant to horizontal flipping
(ΔAUC +0.0001) despite the marked asymmetry of the thorax.

That shortcut is **dataset-specific and does not transfer**: across four external sites the
border-only model scores 0.5209, 0.6082, 0.6240 and 0.5474 against its 0.9094 at home, while
a lungs-only model tracks the full image within 0.015 everywhere. The internal figure is the
contaminated one.

But the external figures are not interchangeable. On **curated case-control** sets of the
kind the field uses for external validation, the full model reaches 0.9382 (Shenzhen), 0.8183
(Montgomery) and 0.7932 (DA/DB). On a **screening cohort** with an Xpert Ultra
reference standard it reaches **0.6151 [0.5853, 0.6452]**, against **0.7827** for a certified
commercial product scored on the same patients. At 90% specificity we detect 12% of cases
where the commercial system detects 45%. The failure is not confined to subtle disease: on
the cases the commercial system finds easy we score 0.6647 against its 0.9602.

The cause is the negative class. Our model assigns a median score of 0.886 to
Xpert-negative screening attendees against 0.953 to cases; a benchmark whose controls are
curated does not prepare a model for a population in which most chests are abnormal.
Finally, a k-nearest-neighbour out-of-distribution gate **does not detect the problem**: the
screening cohort sits at the same measured distance as the site where we score 0.9382, and
the gate passes 99.2% of it.

We conclude that a single "external AUC" is not evidence of deployability, and propose three
cheap measurements that distinguish the cases.

---

## 1. Introduction

Chest radiography is the dominant triage modality for tuberculosis where the disease is most
prevalent, and automated screening is correspondingly well studied. Public benchmarks report
performance that would be transformative if it held: on TBX11K, the field's largest annotated
benchmark, published methods reach 99.1% AUC at 92.1% sensitivity and 96.7% specificity.

The standard response to scepticism about such numbers is external validation, and the
standard external sets in this literature are Shenzhen and Montgomery — two small,
well-curated case-control collections from the US National Library of Medicine. A method that
reports a strong AUC on one of them is generally taken to have demonstrated generalisation.

We show that this is a weak test, by measuring the same frozen model along a ladder:

| rung | example | what it is |
|---|---|---|
| benchmark | TBX11K | curated, lesion-annotated, cases and controls collected separately |
| curated external | Shenzhen, Montgomery, DA/DB | independent site, but still case-control |
| screening cohort | Vietnam | programmatic active case finding, microbiological reference standard for every participant in the library |

Our contributions:

1. **A quantified shortcut** — 0.9094 from the image border on TBX11K (§3) — and evidence it
   is not segmentation leakage.
2. **A dissociation across four external sites** showing the shortcut is dataset-specific
   while lung signal transfers (§4). This gives a *positive* argument for an external result
   that an external AUC alone cannot provide.
3. **The central finding (§5):** performance degrades monotonically along the ladder, and
   collapses at the screening rung — 0.6151 against a commercial comparator's 0.7827 on
   identical patients. We localise the cause to the composition of the negative class.
4. **A negative result about OOD gating (§6):** distance-based gates detect appearance
   shift, not task shift, and pass the cohort on which the model fails.
5. **Three cheap protocol additions** that distinguish these cases (§7).

We make no claim that a particular published system is compromised. Our claim is that the
prevailing evaluation protocol cannot tell these situations apart, and that telling them
apart costs very little.

### 1.1 Relation to prior work, and what we do not claim

**We claim no architectural novelty.** Patch-level aggregation over a frozen
chest-radiograph foundation model for tuberculosis is established prior art.

**RADAR** (MICCAI 2026 satellite, TREAT-MMTB_012) is the closest TB-specific work: a frozen
CXR foundation model, multi-depth patch representations, a learned tuberculosis query
cross-attending over patches, modality- and conditional-adversarial objectives, style
randomisation with prediction consistency, seed and view ensembling, and a higher-resolution
read-out explicitly targeting small focal abnormalities. It subsumes the aggregation change
we measure and additionally *mitigates* the acquisition shortcuts that we only diagnose.

**CheXFound / GLoRI** (IEEE TMI 44(12), 2025) establishes global-plus-local feature
integration as a studied adaptation strategy for CXR foundation models; its primary task
setting differs from ours, and we were unable to retrieve its full text in our environment,
so we cite it as context rather than as a verified comparison.

Our contribution is therefore **evaluative, not architectural**: a controlled measurement of
an aggregation choice under matched training data and seeds with a mathematically equivalent
internal control, placed alongside a documented account of what the resulting numbers can
and cannot support. Where we are behind the prior art — notably in doing nothing to mitigate
the shortcuts we quantify — we say so.

---

## 2. Setup

**Training data.** TBX11K: 11,200 radiographs from hospitals in China — 5,000 healthy, 5,000
sick-but-not-tuberculous, 1,200 tuberculosis confirmed by sputum culture, one image per
individual. We use the core train/val lists (8,400 images, 800 tuberculosis) and re-split
them, since the official test labels are withheld: train 5,941 / development 659 /
calibration 899 / sealed 901.

**External data.** Four sites, none seen in training (verified: maximum cosine similarity
between any external image and any training embedding is 0.8389, no pair above 0.90).

| site | n | TB | reference standard | design |
|---|---|---|---|---|
| Shenzhen, China | 662 | 336 | clinical | case-control, same hospital |
| Montgomery, USA | 138 | 58 | clinical | case-control, same programme |
| DA + DB, India | 176 | 74 | clinical | case-control |
| **Vietnam (Ho Chi Minh City)** [^vn] | **2,708** | **280** | **Xpert MTB/RIF Ultra, all participants in the test library** | **programmatic active case finding** |

[^vn]: Released as Dryad [10.5061/dryad.5tb2rbpcv](https://doi.org/10.5061/dryad.5tb2rbpcv)
    and, on the evidence below, the test library analysed in Codlin AJ, Vo LNQ, Dao TP,
    Forse RJ, Dang HTM, Nguyen LH, Nguyen HB, Dinh LV, Sidney Annerstedt K, Lundin J,
    Lönnroth K, "Comparison of different Lunit INSIGHT CXR software versions when reading
    chest radiographs for tuberculosis," *PLOS Digital Health* 2025;4(4):e0000813,
    [doi:10.1371/journal.pdig.0000813](https://doi.org/10.1371/journal.pdig.0000813).
    **The identification is by statistical reproduction, not patient-ID matching**: our
    copy reproduces all six of that paper's published Lunit AUROCs to two decimal places,
    together with n, prevalence and the previous-TB fraction. The Lunit scores used as a
    comparator throughout this report are that paper's published comparator, not an
    independent benchmark we assembled. Enrolment was **risk-based mobilisation** —
    contacts, adults ≥55 and symptomatic people invited ahead of 127 mobile screening
    events, March 2019 to March 2021 — so it is not unselected consecutive enrolment,
    though it remains a screening population rather than a curated case-control set.

Every site supplies its own controls from the same source, so none can be solved by
recognising an acquisition pipeline — the failure mode that disqualifies the composite
datasets common in this field, in which tuberculosis images are attached to normals from an
unrelated collection.

**Model.** Frozen RAD-DINO (ViT-B/14, 518 px) with a linear head on the L2-normalised pooled
embedding, logit-adjusted loss, development-based early stopping, five seeds. Seed-to-seed
AUC spread is 0.0001. Embeddings are cached, so every ablation costs seconds — which is what
makes this style of dissection practical and is itself part of the argument.

**Preprocessing** is identical at every site: greyscale, `INTER_AREA` resize to 512. DICOM
sources additionally pass through the VOI LUT with a 0.5/99.5 percentile clip. We
deliberately do not histogram-match to the training distribution; tuning preprocessing to a
test set is a soft form of fitting to it.

---

## 3. The classifier reads the border

| what the model can see | development AUC | above-chance AUC retained |
|---|---|---|
| everything | 0.9949 | 100% |
| lung field only | 0.9889 | 98.8% |
| everything except the lungs | 0.9532 | 91.6% |
| lungs dilated 48 px, then erased | 0.9493 | 90.8% |
| **outer 12% frame only** | **0.9094** | **82.7%** |

Masked regions are filled with the per-image mean so the excluded area does not announce
itself. **The outer 12% of the image predicts tuberculosis at 0.9094** with no lung tissue
present.

**It is not segmentation leakage.** Lung segmenters trim the apices and tuberculosis is
classically an upper-lobe disease, so the obvious objection is that "lungs erased" retains
pathology. Dilating the erased region by 48 px — removing the apices and peri-hilar margin
with margin to spare — costs **0.0039**. And the border variant makes no segmentation
assumption at all: its retained region is a fixed rectangle, identical for every image.

Two further observations agree. **Horizontal flipping changes nothing** (ΔAUC +0.0001, CI
[−0.0005, +0.0010]) although the thorax is markedly asymmetric. And **lungs-only is
significantly worse than the full image**, which should not happen if the task were being
solved from lung pathology.

**What the border encodes.** Substituting sick controls for healthy ones costs the border
model 0.0728 — five times the full model's 0.0139 — but it remains at 0.9119, so the signal
is not merely "tuberculosis films were acquired unlike normal films". Two explanations
survive our experiments: a tuberculosis-specific collection route distinct from both control
groups, or a patient-level correlate visible in the frame such as body habitus, since the
outer region contains shoulders and soft tissue and weight loss is a classical feature of
active disease. On the Vietnam cohort, self-reported weight loss separates cases at only
0.5300, which bounds the second explanation without eliminating it.

---

## 4. The shortcut does not transfer; lung signal does

All three rows below use **all negatives** (healthy + sick_non_tb) in training, which is
the configuration in which the shortcut comparison was run. Section 5.0 gives the same
cohorts under the healthy-only configuration adopted later; the dissociation this section
reports is unaffected by that change.

| model | TBX11K | Shenzhen | Montgomery | DA+DB | Vietnam |
|---|---|---|---|---|---|
| full image | 0.9949 | 0.9382 | 0.8183 | 0.7932 | 0.6151 |
| lungs only | 0.9889 | 0.9284 | 0.8554 | 0.7791 | 0.6430 |
| **border only** | **0.9094** | **0.5209** | **0.6082** | **0.6240** | **0.5474** |

At Shenzhen the border model is at chance (CI [0.4759, 0.5642]), retaining 5% of its home
advantage. Across all four sites it never approaches 0.9094. The shortcut is a property of
the TBX11K population: the border weights encode how *that* dataset's collection groups
differ, which is meaningless elsewhere.

The Vietnam cohort makes this decisive. It records the radiography system for every
participant, and **the scanner does not predict the outcome** — TB rates of 10.8%, 11.3% and
8.8% across three machines, with `rad_system` giving AUC 0.4722. Acquisition *cannot* encode
the label there by construction, and the border signal duly disappears.

Meanwhile **lungs-only tracks the full image within 0.015 at every site**, and at the two
most distant sites it is *better* than the full image (Montgomery +0.037, Vietnam +0.028):
discarding everything outside the lung field improves transfer, consistent with the non-lung
weights carrying source-specific content that is neutral nearby and misleading further away.

---

## 5. The ladder, and where it breaks

This is the paper's central result.

| rung | site | AUC | 95% CI |
|---|---|---|---|
| benchmark | TBX11K development | 0.9949 | — |
| curated external | Shenzhen | 0.9382 | [0.9189, 0.9552] |
| curated external | Montgomery | 0.8183 | [0.7308, 0.8986] |
| curated external | DA + DB | 0.7932 | [0.7172, 0.8640] |
| **screening cohort** | **Vietnam** | **0.6151** | **[0.5853, 0.6452]** |
| *same patients, commercial system* | *Lunit INSIGHT 3.9.0.1* | *0.7827* | *[0.7538, 0.8105]* |

### 5.0 The same ladder under the configuration finally adopted

The rungs above are the **full-image, all-negatives** model. Under the configuration this
paper ends up recommending — lungs-only masking, **healthy-only negatives**, regional
12×12 max-pooling — every rung moves, and **the ladder gets steeper, not shallower**:

| rung | cohort | n | TB | matched global | regional max 12×12 |
|---|---|---|---|---|---|
| benchmark | TBX11K development | 659 | 59 | 0.9570 | 0.9368 |
| curated external | Shenzhen | 662 | 336 | 0.9411 | **0.9690** |
| curated external | Montgomery | 138 | 58 | 0.9289 | **0.9690** |
| curated external | **DA** | 72 | 22 | 0.9764 | **0.9755** |
| curated external | **DB** | 104 | 52 | 0.8780 | **0.9497** |
| *curated external* | *DA + DB pooled* | *176* | *74* | *0.9171* | *0.9636* |
| **screening cohort** | **Vietnam REPORT** | **1,356** | **141** | — | **0.7242** |
| *screening cohort, full* | *Vietnam, all* | *2,708* | *280* | *0.6743* | *—* |

Three things must be read carefully here.

**DA and DB are reported separately because they disagree.** Regional max-pooling gains
**+0.0717** [+0.0159, +0.1328] on DB and **−0.0009** [−0.0357, +0.0299] on DA, where the
matched baseline already sits at 0.976. It is **not** the case that regional pooling
improves every cohort: it helps at three sites and is neutral at the fourth.

**The DA+DB pooled row is supplementary.** The two collections have prevalence 0.306 and
0.500 and different score distributions, so pooling them can move AUROC through
cross-cohort ranking effects alone, independently of discrimination within either site. It
is not a single-site result and must not be quoted as one.

**The two Vietnam rows have different denominators** — the reserved REPORT half against the
full cohort — and are not a paired comparison.

Most of the DA/DB movement away from 0.7932 is the **training negative composition**, not
the pooling: refitting the global head with all negatives reproduces the original
lungs-only 0.7791 exactly, and switching to healthy-only negatives alone moves that cohort
by about **+0.138**, roughly three times the +0.0465 pooling adds on top. Healthy-only
negatives remove sick-but-not-TB patients from training, which makes a curated
case-control task easier **without** representing the negatives a screening clinic sends —
which is why the screening rung barely moves. Evidence:
`artifacts/dadb_focal_pooling_20261021_0905/`.

Performance degrades monotonically, and the gap between the benchmark and the clinic is
**0.38 AUC**. Crucially, the commercial comparator shows the screening task is attainable at
0.78 on these very images, so the collapse is a property of our model and not of the data
being impossible.

At the operating points that matter for screening:

| | ours | Lunit |
|---|---|---|
| sensitivity @ 70% specificity | 0.3893 | 0.6857 |
| sensitivity @ 90% specificity | 0.1214 | 0.4464 |
| sensitivity @ 95% specificity | 0.0786 | 0.3500 |

**The failure is not confined to subtle disease.** Splitting cases by how confidently the
commercial system detects them, we score **0.6647 on the obvious half** (where it scores
0.9602) and 0.5656 on the subtle half (0.6052). We miss florid tuberculosis too.

**The cause is the negative class.** Median scores on the screening cohort:

| | cases | non-cases |
|---|---|---|
| ours | 0.9530 | **0.8856** |
| Lunit | 0.8890 | 0.5680 |

Our model calls nearly everything tuberculosis. This follows from a property of the benchmark
that is visible internally once looked for: TB-versus-healthy on TBX11K is **AUC 1.0000**
(interval [1.0000, 1.0000] — a completely saturated metric), while TB-versus-sick is 0.9861.
A screening cohort is the second case and more: its Xpert-negatives include
healed tuberculosis, COPD and other chronic disease, and its cases are screen-detected rather
than the lesion-annotated examples a detection benchmark is built from. The model learned to
recognise abnormality; in a screening queue, abnormality is the base rate.

TBX11K is, to its credit, better than most public tuberculosis datasets in this respect — it
supplies 3,800 sick controls by design, and swapping healthy for sick controls costs our full
model only 0.0139, against reported drops of 0.075–0.306 for models trained on composite
datasets. That is not enough.

---

## 5a. An aggregation change recovers part of the gap, and does not close it

Global average pooling summarises the whole radiograph in one vector. Tuberculosis is focal.
Replacing it with a maximum over a 12×12 grid of region scores, on the same frozen backbone
with the same training data and seeds, reaches **0.7242 [0.6843, 0.7618]** on the reserved
**REPORT half** (n = 1,356, 141 TB), against the commercial comparator's **0.7862** on the
same half — a gap of 0.062.

**The earlier figures on this cohort are not on the same denominator.** The 0.6151
full-image model and the 0.6743 lungs-only global model were both scored on the **full
2,708-participant cohort**, where the comparator scores 0.7827 — a gap of 0.168. A
global-pooling score on the REPORT half alone was never computed, so **0.6151 → 0.7242
cannot be read as a single before/after improvement**: it mixes a configuration change
with a change of evaluation set. What the pre-registration does establish is the
aggregator's effect *within* the REPORT half, via the matched `mean` control below.

Selection was pre-registered: the cohort was split 1,352 / 1,356 before any sweep, the
scoring script was committed before results existed, and **selection optimism was −0.0005**.
An internal control confirms the mechanism — `mean` over the same regions, which is
mathematically global pooling, stays flat at 0.636–0.642 across every grid while `max`
climbs, so the gain is not extra head capacity. Learned gated attention was tried and was
*worse* (0.6416), which we report because it runs against the prior-art design.

On two curated case-control benchmarks the same comparison holds in direction: regional
max exceeds a **matched refitted global baseline** by +0.0279 [+0.0128, +0.0434] on Shenzhen
and +0.0401 [+0.0089, +0.0767] on Montgomery. Those two cohorts previously informed part of
this configuration, so they are exploratory paired comparisons, not independent validation,
and their balanced prevalence (0.51 and 0.42 against the screening cohort's 0.103) makes
their absolute AUROCs incomparable to screening performance.

**This does not close the gap.** 0.7242 against 0.7862 remains a clear deficit, and the
paper's central finding — that benchmark performance does not predict screening performance
— is unchanged by it.

## 6. Distance-based OOD gating does not detect this

A natural defence is a gate that abstains when inputs look unfamiliar. We fitted a k-nearest
neighbour gate on training embeddings and measured its behaviour at each site.

| site | median OOD score | rejected by gate | AUC |
|---|---|---|---|
| TBX11K | 0.3637 | 0.2% | 0.9949 |
| Shenzhen | 0.5331 | 1.2% | 0.9382 |
| **Vietnam** | **0.5312** | **0.8%** | **0.6151** |
| CheXpert (frontal, no TB labels) | 0.7787 | 87.3% | — |

**Vietnam sits at the same measured distance as Shenzhen and performs 0.32 AUC worse.** The
gate passes 99.2% of the cohort on which the model fails.

The reason is structural rather than an implementation defect: a distance-based gate measures
whether images *look like* the training distribution. It cannot see that the *task* has
changed — that the negative class now contains the chronically abnormal, and the positives
are earlier. Appearance shift and task shift are different things, and only the first is
visible to this family of methods.

We had ourselves claimed, on three sites, that this distance predicted transfer. The fourth
site falsified it. We report this because the failure mode is general: a gate that cannot see
a case-mix shift will wave through exactly the deployment that is about to fail.

A separate and more easily fixed error is worth recording. Our gate's threshold was initially
set conformally at α = 0.05 on development, which places it at the 95th percentile of
in-distribution scores and therefore *produces* a 5% false-flag rate whether or not any alien
input lies near that boundary — discarding 9.67% of sick and 3.39% of tuberculosis cases for
no benefit, since alien detection was already saturated at 100% well above it. An upper bound
had been used as a target.

---

## 7. What we propose

Three measurements, none requiring new annotation, which together distinguish the cases this
paper separates.

**1. A region-ablation baseline.** Mask the anatomy of interest, re-run the same head, report
the number. On cached embeddings this costs minutes. If a border-only model approaches the
full-image figure, a large share of the reported performance is not coming from the anatomy —
and evaluating the *same ablation externally* distinguishes a dataset-specific shortcut
(inflates internally, dies externally) from a genuine extra-anatomical signal.

**2. The control-group composition, and a sick-control comparison where the data allows.**
TB-versus-healthy saturates at 1.0000 and is uninformative. Any paper reporting against
healthy-only controls should say so prominently.

**3. At least one consecutive cohort, or an explicit statement that none was tested.** Our
curated external sets overstate screening performance by 0.18–0.32 AUC. "Externally
validated" on a case-control set is a much weaker claim than it sounds, and the difference is
large enough to change a deployment decision.

We would add that **reporting against a commercial comparator on the same patients** was the
single most informative thing we did. It converted "0.6151 seems low" into "0.6151 against
0.7827 on identical images", which is the difference between an ambiguous result and a
diagnosis.

---

## 8. Limitations

One benchmark, four external sites, one of them a screening cohort; a provisional split, with
the sealed one-shot evaluation still to run. Development contains 59 tuberculosis cases, so
sensitivity moves in steps of 1/59. Montgomery (138 images) and DA/DB (176) have wide
intervals. "Border-only" isolates the one shortcut channel we could identify and mask; a
non-pathological cue inside the lung field would not be caught by it. HIV status was
unavailable in every cohort we tested — the Vietnam depositors withheld it — and since
HIV-positive tuberculosis presents atypically, performance in that population is unmeasured
and plausibly worse. We did not attempt to fix the model; whether fine-tuning, a different
negative-class composition, or a larger training distribution closes the 0.38 gap is open, and
the commercial comparator's 0.7827 shows that something can.

---

## 9. Reproducibility

Every figure is produced by a script in the project repository with its artifact retained.
The backbone is public with its revision pinned; the benchmark is public; all four external
sets are public, three of them downloadable without any agreement. The ablations run in
minutes on a 6 GB laptop GPU.

---

## Notes for the team (not part of the submission)

- **Everything is on the provisional split.** Regenerate once C01 is frozen; the scripts take
  minutes and the prose should survive.
- **The sealed external evaluation has not run.** §5 is exploratory and a reviewer will ask.
- **Held back deliberately**, each a separate note or report section: the latent-versus-active
  indistinguishability (AUC 0.55, with 21% of positive calls being healed disease), the
  backbone comparison against DenseNet-121, the noise-induced specificity collapse from 95%
  to 26%, and the TBX11K archive findings (158 cross-split duplicates; 576 non-benchmark
  images reachable via `all_*.txt`; 12.1% of the official test split being relabelled public
  images).
- **D09 should be reopened.** Lungs-only beats the full image at the two most distant sites.
  The crop costs ~0.01 at home and gains 0.037 and 0.028 where it matters.
- **Authorship and scope** need agreeing before submission; §6's dataset findings touch P1's
  workstream.
