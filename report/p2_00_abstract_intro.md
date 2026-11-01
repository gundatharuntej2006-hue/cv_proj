# Abstract, Introduction and Literature Review

*Owner: Person 2 (assigned §abstract, §introduction, §literature, §conclusion). Numbers are
provisional pending the frozen split; every figure cites an artifact.*

---

## Abstract

Tuberculosis remains one of the leading infectious causes of death, and chest radiography is
the primary triage tool in the settings where it is most prevalent. We built an explainable
screening system — a quality and out-of-distribution gate, a classifier, lesion localisation,
calibrated three-way triage, and a reviewing dashboard — and then spent most of our effort
asking whether its headline number meant anything.

On TBX11K, the largest public tuberculosis benchmark, a frozen RAD-DINO backbone with a
linear probe reaches **AUC 0.9949**. We show this figure is inflated: the same model predicts
tuberculosis at **0.9094 from the outer 12% of the image**, a region containing no lung
tissue, and is unaffected by horizontally flipping an anatomically asymmetric organ. That
shortcut is specific to the benchmark — across four independent sites it falls to 0.52–0.62 —
and a lungs-only model carries the external result instead, which is the evidence that the
external numbers reflect pathology.

Those external numbers, however, are not interchangeable. On curated case-control sets of the
kind conventionally used for external validation we reach 0.9382 (Shenzhen, China), 0.8183
(Montgomery, USA) and 0.7932 (DA/DB, India). On a **screening cohort** of 2,708[^cohort]
participants with an Xpert Ultra reference standard we reach **0.6151**, against **0.7827**
for a certified commercial system scored on identical images. At 90% specificity we detect
12% of cases where it detects 45%. The failure traces to the negative class: our model
assigns a median score of 0.886 to screening-cohort negatives, because a benchmark with
curated controls does not prepare a model for a population in which most chests are abnormal.
A nearest-neighbour out-of-distribution gate does not detect the problem, passing 99.2% of
the cohort on which the system fails.

> **Configuration note (2026-10-21).** The five figures above are one internally consistent
> configuration: full-image input, global pooling, **all negatives** (healthy + sick_non_tb).
> Under the configuration later adopted — lungs-only masking, **healthy-only negatives**,
> regional 12×12 max-pooling — the curated sites rise substantially (Shenzhen 0.9690,
> Montgomery 0.9690, DA 0.9755, DB 0.9497) while the screening cohort reaches only **0.7242**
> (Vietnam REPORT, n = 1,356, 141 TB). **The ladder is steeper under the newer configuration,
> not shallower**, so the argument of this paper is unaffected. Most of the DA/DB movement is
> the change of training negatives rather than the pooling: refitting with all negatives
> reproduces the original 0.7791 exactly, and healthy-only negatives alone move that cohort
> by about +0.138. Healthy-only negatives make a curated case-control task easier without
> representing the negatives a screening clinic sends. Per-cohort figures, intervals and
> provenance: `docs/p2_external_validation.md` and
> `artifacts/dadb_focal_pooling_20261021_0905/`.

We report the engineering as specified and working — the gate rejects 100% of non-chest and
lateral inputs at a 0.2% false-flag cost, the runtime reproduces exported scores, and
held-out evaluation is enforced by a one-shot sealing procedure — but we do not claim the
system is deployable. Our contribution is the measurement: a single external AUC is not
evidence of deployability, and three cheap additions to the standard protocol distinguish the
cases.

---

## 1. Introduction

### 1.1 The problem

Tuberculosis killed roughly 1.25 million people in 2023 and remains the leading cause of
death from a single infectious agent outside pandemic years. Case-finding is the bottleneck:
a substantial fraction of cases are never diagnosed, and the WHO recommends chest radiography
as a triage test ahead of confirmatory molecular testing precisely because radiography is
cheap, fast and widely available while Xpert cartridges and laboratory capacity are not.

That makes automated chest-radiograph triage an attractive target, and the literature
reflects it. Reported performance on public benchmarks is high enough that the task can look
solved. The question this project set out to answer is whether it is.

### 1.2 What we built

A four-person system, of which this report section covers the whole:

- **Data, splits and leakage control** (Person 1) — standardised images, lung regions,
  patient-level splits, duplicate control.
- **Quality and out-of-distribution gate, and classification** (Person 2) — reject what
  cannot be assessed; classify what can.
- **Lesion localisation and explainability** (Person 3) — bounding boxes and heatmaps, so a
  reviewer sees why.
- **Calibration, triage and evaluation protocol** (Person 4) — calibrated probabilities,
  a three-way TB / REFER / NOT-TB decision, and the evaluation discipline.

The design commitments that shaped the work were made early and held: a **frozen** backbone,
so that every experiment is cheap and exactly reproducible on a 6 GB laptop; a **gate that
abstains** rather than guesses; and **held-out data read exactly once**, enforced in code
rather than promised in prose.

### 1.3 What we found, and why the report is shaped this way

The system works as specified. It also reaches 0.9949 on the benchmark, and most of this
report is about why we do not lead with that number.

Three findings, in increasing order of importance:

**The headline figure is partly an artifact.** Masking experiments show the classifier
predicts tuberculosis at 0.9094 from the image border alone. We trace this to the benchmark's
construction — cases and controls were collected as separate groups, so acquisition
correlates with the label — and we show it does not survive a change of hospital.

**The out-of-distribution gate was mis-thresholded, and its guarantee is narrower than
believed.** A conformal threshold used as a target rather than a bound was discarding 9.67%
of sick patients for no detection benefit. More fundamentally, distance-based gating detects
whether images *look* unfamiliar, not whether the task has changed.

**Benchmark performance does not predict screening performance.** Moving from the benchmark
to curated external sets to a screening cohort, AUC falls 0.9949 → 0.94–0.79 →
0.6151, while a commercial system reaches 0.7827 on the same patients. This is the project's
central result and it is a negative one.

We report it this way deliberately. A student project that reports 0.99 and stops has
produced a number; one that finds out why the number is wrong has produced a finding. The
second is more useful, and it is also more honest about what would happen if anyone installed
this system in a clinic.

### 1.4 Scope and limitations of the claim

We do not claim a deployable tuberculosis screening tool, and nothing in this report should
be read as clinical validation. We claim: a working, reproducible pipeline; a quantified
account of what its benchmark performance is made of; and measurements that the evaluation
protocol standard in this area would not have surfaced.

---

## 2. Literature review

### 2.1 Automated tuberculosis detection from chest radiographs

Computer-aided tuberculosis detection predates deep learning, but the modern line begins with
the release of the two NLM collections — **Shenzhen** (662 images) and **Montgomery** (138) —
by Jaeger et al., which for a decade were the de facto benchmark. Their size is their
limitation: 394 cases between them cannot support a confidence interval tight enough to
separate competing methods, and both are curated case-control collections rather than
consecutive cohorts.

**TBX11K** (Liu et al., CVPR 2020; extended in TPAMI) changed the scale of the field: 11,200
radiographs from Chinese hospitals with **culture-confirmed** labels, one image per
individual, and — unusually — *lesion bounding boxes* and a dedicated 5,000-image
sick-but-not-tuberculous class. The accompanying **SymFormer** architecture, which exploits
the bilateral symmetry of the thorax and performs classification and detection jointly,
reports 99.1% AUC with 92.1% sensitivity at 96.7% specificity, and remains the state of the
art.

Two properties of TBX11K deserve more attention than they receive. Its sick-non-TB class is
genuinely valuable: most public tuberculosis datasets are assembled by attaching a
tuberculosis folder to a set of healthy normals from an unrelated source, which permits a
model to score highly by recognising the acquisition pipeline. We confirm the value
empirically — swapping healthy for sick controls costs our model only 0.0139, against
published drops of 0.075–0.306 elsewhere. Conversely, its *distributed archive* contains
properties that affect reproducibility and are not documented: we find 158 near-duplicate
pairs spanning the official split boundaries, 576 images from external datasets reachable
through one of the two shipped list files, and 400 of the 3,302 held-out test images being
copies of publicly labelled radiographs.

### 2.2 Foundation models for radiology

The dominant recent approach is transfer from a large pretrained backbone. **RAD-DINO**
(Pérez-García et al.) applies DINOv2 self-supervision to chest radiographs alone, without
paired text, and reports that image-only pretraining matches or beats language-supervised
alternatives on segmentation and classification. **TorchXRayVision** packages conventional
supervised DenseNet models trained across several hundred thousand labelled radiographs.

We compare the two directly. With an identical linear head, identical seeds, and byte-identical
inputs, RAD-DINO reaches 0.9949 against 0.9539 for a DenseNet-121 (ΔAUC +0.0409, CI excluding
zero), and an off-the-shelf zero-shot baseline reaches 0.8026. The backbone choice is, on this
benchmark, the only pipeline decision that measurably matters — crop margin, k in the OOD
gate, and binary-versus-three-class heads make no measurable difference.

### 2.3 Shortcut learning in medical imaging

The concern that medical imaging models exploit non-pathological correlates is
well-established. Models have been shown to infer scanner, hospital and even patient
demographics from radiographs, and to use those signals when they correlate with the label.
Zech et al. demonstrated the canonical case, where pneumonia classifiers partly learned the
portable-scanner marker.

What is less established is a cheap protocol for *detecting* the problem in a given result.
Our region-ablation measurement (§3 of the methodology) is deliberately simple — mask the
anatomy, re-run the same head — and the key addition is to evaluate the ablation *externally*,
which separates a dataset-specific shortcut from a genuine extra-anatomical signal. A recent
audit of medical vision-language models reports the complementary finding from the
control-group side: substituting sick for healthy controls dropped AUROC by 0.075–0.306 across
every model tested, and one supervised model fell from 0.999 internally to 0.629 externally.

### 2.4 Out-of-distribution detection and its limits

Nearest-neighbour distance in a learned embedding space is a standard and effective
out-of-distribution detector, often combined with conformal prediction to give a
distribution-free bound on the false-flag rate. Our gate follows this design, and on the task
it was built for it works: 100% of non-chest and lateral-view inputs rejected at a 0.2%
in-distribution false-flag cost.

Its limits are the more interesting result, and both are, we believe, general. First,
conformal calibration bounds the false-flag rate but does not *choose* a good threshold: used
as a target it places the threshold at the 95th percentile of in-distribution scores and
therefore produces a 5% rejection rate whether or not any alien input lies nearby. Second,
and more fundamentally, appearance distance does not detect task shift — our screening cohort
sits at the same measured distance as a site where we score 0.9382, and we score 0.6151 on it.

### 2.5 The evaluation gap

The literature contains a recognised gap between retrospective benchmark performance and
prospective screening performance, and the WHO's own evaluations of commercial CAD products
report figures in the 0.75–0.90 range on screening populations — well below benchmark
numbers. Our work connects the two ends with a single frozen model measured along the whole
ladder, including against a commercial product on identical images, which to our knowledge
has not been reported for an academic system.

---

*Section sources: `docs/p2_shortcut_probe.md`, `p2_shortcut_does_not_transfer.md`,
`p2_vietnam_result.md`, `p2_external_validation.md`, `p2_control_group.md`,
`p2_backbone_baseline.md`, `p2_ood_threshold_is_miscalibrated.md`,
`p2_tbx11k_composition_corrected.md`, `p2_duplicate_finding.md`. Citations to be formalised
at assembly; claims attributed to published work above are from the TBX11K/SymFormer papers,
the RAD-DINO paper, TorchXRayVision, Jaeger et al. on the NLM sets, Zech et al. on shortcut
learning, and the 2026 VLM audit — exact references to be inserted in §references.*

[^cohort]: The screening cohort is the test library released as Dryad
    [10.5061/dryad.5tb2rbpcv](https://doi.org/10.5061/dryad.5tb2rbpcv) and, on the
    evidence recorded in `docs/p2_vietnam_result.md`, the dataset analysed in Codlin AJ,
    Vo LNQ, Dao TP, Forse RJ, Dang HTM, Nguyen LH, Nguyen HB, Dinh LV, Sidney Annerstedt K,
    Lundin J, Lönnroth K, "Comparison of different Lunit INSIGHT CXR software versions when
    reading chest radiographs for tuberculosis," *PLOS Digital Health* 2025;4(4):e0000813,
    [doi:10.1371/journal.pdig.0000813](https://doi.org/10.1371/journal.pdig.0000813).
    Identified by statistical reproduction rather than patient-ID matching. The Lunit
    comparator scores quoted throughout are that paper's published comparator. Enrolment
    was risk-based mobilisation ahead of 127 mobile screening events, not unselected
    consecutive enrolment.
