# Near parity, not superiority: what a frozen foundation model can and cannot do for tuberculosis screening on a real cohort

**Draft v2.0 — 2026-11-01 — all TBX11K numbers on the provisional split; every Vietnam number on an exposed cohort (see §7)**

*Target: workshop paper (ML4H / MICCAI workshop) or arXiv preprint, ~6–8 pages in IEEE two-column. Author order and scope to be agreed by the team before submission.*

*Authors: Tharun [P1], Atul Dhull [P2], Ritika [P3], Kunal [P4] — Department of AI & ML, BMS Institute of Technology and Management, Bengaluru.*

---

## Abstract

Published methods report above 0.99 AUROC on TBX11K, the largest annotated tuberculosis chest-radiograph benchmark. We measure what survives when the same frozen model is evaluated on a programmatic Vietnamese active-case-finding cohort with a microbiological reference standard and a commercial CAD comparator evaluated on the same patients, and we report every attempt to close the gap, including the ones that failed.

A frozen RAD-DINO encoder (86.6 M parameters) with a regional head of 769 trainable parameters per seed (3,845 across five seeds; max over a 12×12 grid of region logits) reaches 0.9368 on TBX11K development and **0.7242 [0.6843, 0.7618]** on the reserved half of a 2,708-participant Vietnamese screening cohort, against the comparator's 0.7862. The deficit concentrates in participants with previous tuberculosis (0.7720 → 0.6392), reproducing a published effect; whether we degrade *more* than the comparator is not statistically supported (+0.048 [−0.017, +0.111]). Scanner-subspace removal (−0.0002), head refitting on local labels (0 of 30 comparisons), lesion-box supervision (+0.0019 [−0.0019, +0.0061]) and encoder ensembling (+0.0006 [−0.0067, +0.0076]) are each a null on their declared primary endpoint; multi-source training is a real but sub-threshold +0.0033 [+0.0006, +0.0064]. Three changes help on the screening cohort and are invisible on the benchmark — training to the active-TB label (+0.020), a related encoder (+0.019), and a twelve-field screening questionnaire fused by logistic regression (+0.058) — and they overlap rather than add. The best configuration reaches 0.7941 fused on the tuning half, numerically above the comparator's 0.7780 with an interval that includes zero; a field ablation shows the part above the comparator is a two-column scanner-associated adjustment, which unlabelled per-scanner centering recovers (0.7795 with 25 images per scanner; paired +0.0281 [+0.0074, +0.0486] at 100) under the nominal sampling condition tested, and which fails a pre-registered stability bound under population shift. A concept-probing instrument built on radiology-report embeddings was found to be negation-blind and its apparent findings are withdrawn. The decision region of the head overlaps an annotated active-TB lesion in 95.9% [89.8, 100] of 49 development images.

The candidate achieved a numerically higher AUROC point estimate than the comparator on the exposed tuning cohort; the paired confidence interval included zero, and no formal superiority, equivalence or non-inferiority claim is supported. The system has a measured low-memory inference footprint on a consumer laptop GPU, exact attribution and a measured localisation result. We give the protocol additions that made these distinctions visible and the reasons no superiority claim is available from public data.

---

## 1. Introduction

Chest radiography is the dominant triage modality for tuberculosis where the disease is most prevalent, and automated reading is correspondingly well studied. On TBX11K, purpose-built architectures report 99.1% AUROC [1], and general-purpose vision–language models reach about 0.80 zero-shot on the same benchmark [11]. The standard response to scepticism is external validation on Shenzhen and Montgomery [2], two small curated case-control collections, and a strong result there is generally taken as evidence of generalisation.

We evaluated one frozen model along a ladder — benchmark, curated external sites, a programmatic screening cohort with Xpert MTB/RIF Ultra reference for every participant and a commercial reader scored on the same images [3] — and then spent the rest of the project trying to close the gap that appeared at the last rung. This paper reports both halves. It makes no architectural claim: patch-level aggregation over a frozen chest-radiograph foundation model is established prior art [4, 5]. Its contributions are evaluative and methodological:

1. **The ladder.** The same regional model loses approximately 0.21 AUROC between TBX11K development and the Vietnamese screening cohort, illustrating the substantial difference between curated benchmark evaluation and programmatic screening evaluation (§3). A quantified border-reading shortcut explains part of the benchmark figure and does not transfer (§3.1).
2. **Where the loss is.** It concentrates in previous tuberculosis, a published effect [3, 6, 7] that we reproduce rather than discover; removing the scanner-predictive embedding subspace did not improve screening AUROC, although scanner-associated score offsets later proved to matter for pooled ranking (§4, §5.5).
3. **Eleven interventions, pre-registered, evaluated against their declared decision criteria** — several negligible or adverse, one positive but sub-threshold on the primary split, three positive on the exposed screening cohort (which does not constitute independent validation), and the finding that those three overlap (§5).
4. **Clinical-context fusion** that benefits our score and not the comparator's, decomposed into a scanner-associated adjustment and a diffuse questionnaire contribution (§5.4–5.5), with a label-free calibration method that worked under the nominal sampling condition tested and failed under population shift (§5.6).
5. **A negative result on concept probing**: text-anchored concept scores from a radiology vision–language model cannot separate a finding from its negation, and conclusions drawn from them are withdrawn (§6).
6. **An exact-attribution result**: the decision region of a max-pooled head is the region that decided, and it overlaps an annotated active lesion 2.6× as often as a random lung region (§5.3).

We also report what we did *not* achieve, because the field's incentives run the other way: nothing here establishes superiority over the commercial comparator, and §7 explains why no evaluation on public data could.

---

## 2. Data and model

**TBX11K** [1]: 11,200 radiographs from Chinese hospitals, one per individual; we use the 8,400 core train/val images (800 tuberculosis, with lesion boxes labelled *ActiveTuberculosis* or *ObsoletePulmonaryTuberculosis*) and re-split them, since official test labels are withheld: train 5,941 / development 659 / calibration 899 / internal test 901 (provisional split C01; the internal test partition was scored once during a rehearsal and is not used here). Training rows are healthy plus tuberculosis (3,241 images, 541 positive under `tb_label`, 447 under `active_tb_label`); the 2,700 sick-non-TB train images are excluded for the production head (see §3.2 for why that matters).

**Vietnam screening cohort** [3]: 2,708 participants from 127 mobile active-case-finding events in Ho Chi Minh City, 2019–2021, with Xpert Ultra for every participant, three radiography systems, a twelve-field screening questionnaire, and Lunit INSIGHT CXR 3.9.0.1 scores. 280 are Xpert-positive; 36% of participants report previous tuberculosis. The cohort was split once, before any tuning, stratified by outcome and scanner, into a tuning half (TUNE, 1,352, 139 positive) and a reporting half (REPORT, 1,356, 141 positive). The published AUROCs of [3] reproduce from our copy to two decimals, which is how we identified it as that paper's test library. Enrolment was risk-based mobilisation, not unselected consecutive enrolment.

**Curated external sites**: Shenzhen (662, 336 TB), Montgomery (138, 58 TB), and the TBX11K DA/DB lists (176, 74 TB).

**Model.** RAD-DINO ViT-B/14 [8], revision pinned, frozen (86,580,480 parameters). Images are standardised to 512 px, lung-masked with a public segmenter (outside-lung pixels set to the image mean), encoded at 518 px, and the 37×37 patch tokens are average-pooled to a **12×12 grid of 768-d L2-normalised region vectors**. The head is one `Linear(768→1)` applied to every region; the image logit is the **maximum over the 144 region logits**. Training: AdamW 3e-3, batch 256, weight decay 1e-4, logit adjustment by the training prior, early stopping on development AUROC, seeds 17/29/43/61/79, scores averaged over seeds. Trainable parameters: 769 per seed, 3,845 in total; the optional fusion layer (§5.4) adds 13.

**Baseline model versus research candidate.** Two configurations appear in this paper. The *baseline* is the production head described above. The *research candidate (v3)* is the configuration behind the 0.7941 result of §5.4; it is frozen and not promoted.

| component | baseline | research candidate v3 |
|---|---|---|
| encoder (frozen) | `microsoft/rad-dino` @ `110cbc18d5133582e320b43d53bf5c44e410c936` | `microsoft/rad-dino-maira-2` @ `a6132efb9268859545e9f68be18a76a6473f174a` (same ViT-B/14 architecture, continued self-supervised training) |
| training target | `tb_label`: active, latent and mixed TB positive; healthy negative | `active_tb_label`: active and mixed TB positive; latent (obsolete-only) TB **relabelled negative** alongside healthy |
| training rows | 3,241 (541 positive) | 3,241 (447 positive) |
| region aggregation | 12×12 maximum | 12×12 maximum |
| head | `Linear(768→1)`, 769 parameters per seed, 5 seeds | identical |
| clinical fusion | optional, 12 fields | optional, 12 fields including two scanner indicators |
| status | served default (bundle v2) | frozen research candidate (bundle v3), not promoted |

**Statistics.** Paired bootstrap on shared indices (2,000–4,000 resamples, fixed seed); every intervention carried a protocol hashed before fitting, a minimum meaningful effect of +0.02 with the interval excluding zero, and a reproduction gate requiring the production arm to be reproduced to 0.0005 before any contrast was read. Resampling is at the participant level (one image per participant). The released data carry no screening-event identifier, so clustering of participants within the 127 events cannot be modelled; if outcomes cluster within events, participant-level intervals understate uncertainty.

---

## 3. The ladder

| rung | cohort | n | TB | global head (full image) | regional 12×12 max |
|---|---|---|---|---|---|
| benchmark | TBX11K development | 659 | 59 | 0.9949 | 0.9368 |
| curated external | Shenzhen | 662 | 336 | 0.9382 | 0.9690 |
| curated external | Montgomery | 138 | 58 | 0.8183 | 0.9690 |
| curated external | DA / DB | 72 / 104 | 22 / 52 | 0.7932 (pooled) | 0.9755 / 0.9497 |
| **screening** | **Vietnam REPORT** | **1,356** | **141** | — | **0.7242 [0.6843, 0.7618]** |
| *same patients, commercial* | *Lunit 3.9.0.1* | | | | *0.7862* |

The global row is the earlier full-image probe on the full 2,708 cohort (0.6151 against the comparator's 0.7827 there); the regional row is on the reserved half. They are different denominators and are not a before/after pair. What was pre-registered is the aggregator's effect within the tuning half: `mean` over the same regions — mathematically global pooling — stays at 0.636–0.642 across every grid size while `max` reaches 0.7239, and a learned gated-attention aggregator is worse (0.6416). Selection optimism between the tuning and reporting halves is −0.0005.

### 3.1 The benchmark figure is partly a border

The global probe reaches **0.9094 from the outer 12% of the image alone** — shoulders, collimation, background, no lung — so 82.7% of its above-chance performance is available without the anatomy of interest. Dilating the erased lung region by 48 px costs 0.0039; horizontal flipping changes nothing (+0.0001). The shortcut does not transfer: the border-only model scores 0.52–0.62 at the four external sites, while a lungs-only model tracks the full image within 0.015 everywhere and beats it at the two most distant sites. The internal figure is the contaminated one, which is why the production model is lung-masked.

### 3.2 The screening failure is the negative class

TB-versus-healthy on TBX11K is AUROC 1.0000; TB-versus-sick is 0.9861. A screening queue is the second case and worse: its Xpert-negatives include healed tuberculosis and chronic lung disease. Median scores on the cohort are 0.953 for cases and 0.886 for non-cases under the global probe — the model learned abnormality, and in a screening queue abnormality is the base rate. A distance-based out-of-distribution gate does not detect this: the cohort sits at the same measured distance as Shenzhen (median 0.531 vs 0.533) and the gate passes 99.2% of it.

---

## 4. Where the loss is

**Previous tuberculosis.** Under the construction used by the published literature (active TB vs Xpert-negative, within history strata), our regional head scores 0.7720 [0.740, 0.803] in participants without previous TB and **0.6392 [0.586, 0.692]** with it — degradation **+0.1328 [0.0718, 0.1951]**. The comparator degrades too (0.8170 → 0.7192). The difference in differences, **+0.0481 [−0.0170, +0.1113]**, includes zero: the claim that we degrade *more* than the comparator is not supported. The effect itself was published on this very cohort by its depositors [3] and earlier by others [6, 7]; we reproduce it. `past_tb` is self-reported history, not a radiological annotation, and is treated as such throughout.

**Scanner-predictive embedding directions.** Removing the 44 of 768 embedding directions that predict the radiography system did not improve screening AUROC (−0.0002 [−0.0009, +0.0004]) while scanner decodability stayed at 0.91. Subsequent experiments nevertheless identified scanner-associated score *offsets* that materially affect pooled ranking and operating-point behaviour (§5.5–5.6); the two findings are different and both stand.

---

## 5. Interventions

Every row below was a pre-registered run with a hashed protocol; the primary split is TBX11K calibration (899; 81 active-TB positives, scored against the active label); the screening tuning half is reported as context and is exposed. Rows 1, 2, 4, 6 and 11 act on cohort-specific quantities (scanner subspace, aggregator selection, local-site refit, questionnaire fusion, per-scanner centering) and by design have no calibration endpoint; their declared criteria were on the screening cohort (row 1 on REPORT, before the exposure audit) and they are exploratory. Rows 3, 5 and 7–10 have the calibration endpoint as primary.

| # | intervention | calibration Δ (primary) | screening TUNE Δ | verdict |
|---|---|---|---|---|
| 1 | scanner-subspace removal (INLP) | — | −0.0002 [−0.0009, +0.0004] (REPORT) | null |
| 2 | gated-attention MIL instead of max | — | −0.082 | worse |
| 3 | **active-TB label** (latent → negative) | −0.0037 [−0.0081, +0.0002] | **+0.0203 [+0.0127, +0.0279]** | external only |
| 4 | head refit on local labels vs threshold only | 0 of 30 comparisons help; max +0.0031 | — | null |
| 5 | lesion-box region supervision (λ = 1) | +0.0019 [−0.0019, +0.0061] | −0.0108 [−0.0182, −0.0037] | null / worse |
| 6 | **clinical-context fusion** (12 fields, LR) | — | **+0.058 [+0.030, +0.085]** | external only |
| 7 | three training sources (+ Shenzhen, Montgomery) | +0.0033 [+0.0006, +0.0064] | +0.0036 | real, sub-threshold |
| 8 | encoder ensemble (RAD-DINO + MAIRA-2 + XraySigLIP) | +0.0006 [−0.0067, +0.0076] | +0.011 | null |
| 9 | **RAD-DINO-MAIRA-2 as drop-in encoder** | −0.0004 [−0.0076, +0.0067] | **+0.0188 [+0.0015, +0.0364]** | external only |
| 10 | levers 3 + 7 + 9 stacked | −0.0010 [−0.0099, +0.0083] | +0.0286 [+0.0089, +0.0490] (predicted +0.047) | not additive |
| 11 | label-free per-scanner centering | — | +0.0281 [+0.0074, +0.0486] (n = 100) | worked under nominal sampling; fails stability bound |

### 5.1 Three levers, invisible internally, overlapping externally

The active-TB label, the related encoder and the extra sources each leave calibration unchanged and move the screening cohort. The label effect is a *relabelling* effect: dropping the latent images instead is worth +0.0023. The three are not additive — the stack recovers +0.029 of a predicted +0.047, and once the encoder is MAIRA-2 the extra sources subtract (−0.0062 [−0.0124, +0.0001]). The best image-only head on the screening cohort is MAIRA-2 with the active label and TBX11K alone: **0.7585 [0.722, 0.796]**, +0.0348 [+0.0160, +0.0542] over production, −0.0032 on calibration.

### 5.2 Capacity transfers worse

A 2,304-d head on the concatenated features of three encoders has the best development AUROC of anything fitted (0.9505) and loses on the screening cohort (−0.0137 [−0.0324, +0.0041]). The pattern "more internal, less external" held across every configuration in the project — global heads 0.957–0.995 internally and 0.67 externally; attention 0.9495 / 0.6416; regional 0.9368 / 0.7242 — and is our reason for not fine-tuning the backbone.

### 5.3 Attribution is exact, and it lands on lesions

Because the image logit *is* the maximum region logit, the arg-max region is the decision region with no approximation. Attribution is exact at the **12×12 region level** (cells of ≈ 43 px at 512 px), not at pixel-level lesion boundaries. The spatial unit of evaluation is the pooled **12×12 region cell**, not the original 37×37 ViT patch: a cell counts as overlapping a box if the cell's pixel footprint shares at least one 14-px ViT patch with the box's footprint after the exact preprocessing chain (box → 512 → 518 px → patch grid → pooling bins). On the 49 development images carrying an active-lesion box, the production head's arg-max cell overlaps a box — a **top-1 region-cell hit** — in **95.9% [89.8, 100.0]** (top-3 cells: 100%), against 36.7% for a random lung cell, 18.4% for a random cell, 8.2% for the centre cell, and a mean box coverage of 19.7% of cells. The box-supervised head (§5, row 5) has a top-1 region-cell hit rate of 0.939 on the same images — no improvement. This is "the strongest region is on a lesion", not a detector; it is exploratory, on 49 curated images with large boxes.

### 5.4 Clinical-context fusion helps us and not the comparator

A logistic regression (C = 1) on the standardised image score plus twelve questionnaire and acquisition fields (sex, age group, cough, fever, weight loss, night sweats, contact, previous TB, diabetes and unknown-diabetes, two scanner indicators), 10-fold × 20 repeats on the tuning half:

| image score | raw | fused | Δ | previous TB raw → fused |
|---|---|---|---|---|
| production head | 0.7237 | 0.7809 | +0.058 [+0.030, +0.085] | 0.6144 → 0.7122 |
| MAIRA-2 + active head | 0.7585 | **0.7941** | +0.037 [+0.015, +0.058] | 0.6501 → 0.7349 |
| comparator | 0.7780 | 0.7778 | +0.001 [−0.018, +0.019] | 0.7062 → 0.7144 |

Fused MAIRA-2 vs comparator raw: **+0.0170 [−0.0193, +0.0527]**; in previous TB +0.030 [−0.046, +0.107]. The previous-TB improvement under fusion should not be read as the questionnaire repairing the previous-TB deficit of §4: §5.5 shows that in that subgroup the scanner term carries most of it and the `past_tb` field almost none. Sensitivity at 70% specificity: 0.741 vs 0.708. The comparator keeps the better AUPRC (0.328 vs 0.300). We state only what was measured: the comparator's score gains nothing from these twelve fields in this experiment; we do not claim it cannot use clinical context.

### 5.5 The gain decomposes into a scanner-associated adjustment and a questionnaire

Leave-one-field-out on the fused MAIRA-2 model: the only field whose removal costs with an interval excluding zero is the **scanner identifier** (−0.0138 [−0.0250, −0.0029]; −0.0429 [−0.0702, −0.0180] in previous TB). Removing `past_tb` changes the previous-TB AUROC by +0.004. With the ten questionnaire fields and no scanner term the fused model is **0.7803** against the comparator's raw 0.7780, a difference within noise, and image + scanner alone is 0.7788. The point estimate above the comparator is a per-scanner intercept. Scanner indicators produced a substantial improvement in our candidate (≈ 0.02 overall, 0.04 in previous TB), while adding the same fields to the comparator produced no material improvement in this experiment (+0.001 [−0.018, +0.019]). This does not independently establish scanner invariance in either system — that would need scanner-stratified performance and calibration analysis including an unseen scanner; it says that a scanner-associated adjustment is available to our score and was not, in this experiment, to the comparator's.

### 5.6 The scanner offset can be removed without labels — under a condition

On the tuning half the raw candidate's score sits 1.3 logits higher on one of the three scanners at similar prevalence, and the pooled AUROC (0.7546 on a patient-disjoint evaluation group of 677 participants, 70 Xpert-positive; the calibration pools hold the other 675) is 0.03–0.07 below every scanner's own. Centering each scanner's scores on an **unlabelled** batch of its own images, with calibration and evaluation groups disjoint:

| n per scanner | supervised scanner term | unlabelled centering |
|---|---|---|
| 25 | 0.7667 [0.697, 0.784] | **0.7795 [0.769, 0.786]** |
| 100 | 0.7786 | 0.7818 |
| Δ vs raw at n = 100 (paired) | +0.0281 [+0.0070, +0.0484] | **+0.0281 [+0.0074, +0.0486]**; previous TB +0.0557 [+0.0065, +0.1004] |

Within-scanner AUROC is identical under every arm; the effect is pooled ranking and the operating point. Referral burden at a label-free top-30% rule moves from 31% / 24% / 50% across scanners to 40% / 31% / 32%.

The failure mode, pre-registered and found: a calibration batch drawn from previous-TB patients only lowers that scanner's sensitivity from 0.677 to 0.555 (move 0.122; bound 0.10), because centering removes population differences as readily as acquisition ones. The supervised term fails worse in the opposite direction (sensitivity 0.18 on a negatives-only batch; 46% referral burden on an enriched one). By the rule written before the numbers the method is **not usable as is**; it needs a consecutive-sample calibration batch and a fail-closed guard whose safety would have to be established prospectively. The positive result holds under the nominal sampling condition tested — random draws from each scanner's pool — and is not a general representative-sampling guarantee.

### 5.7 Cost

Encoder throughput on a 6 GB laptop GPU: 0.030 s/image (fp16), peak 0.69 GB; 1.25 s/image on an 8-thread CPU. Head fit on 3,241 cached images: 6.6 s. With the lung segmenter on the same GPU (float32; masks agree with the CPU-built training cache at 99.998% of pixels, IoU ≥ 0.9999 on 16 images) the full pipeline runs at a median 0.13 s/image, p90 0.15 s, after a ~40 s first-call model load; on CPU the segmenter alone is ≈ 2.1 s/image. Threshold recalibration to a new site reaches the oracle operating point with 25 local labelled images; head refitting adds nothing to it.

---

## 6. A negative result on concept probing

To ask *what* the score responds to, we projected the cohort through a public radiology vision–language model [9] and correlated the TB score with similarity to single-phrase text anchors (*scarring*, *fibrosis*, *consolidation*, …), using only exact vocabulary entries and excluding concepts without one. Among 1,213 Xpert-negatives, *scarring* correlates with our score at ρ = 0.566 while carrying no information about TB (AUROC vs Xpert 0.494); *consolidation*, the canonical active-TB finding, correlates *negatively* (−0.300).

A forensic pass explained the anomaly and disqualified the instrument. The anchors cannot distinguish a finding from its negation: the text embedding of *no scarring* sits at cosine 0.906 to *scarring*, and the image-side correlation of *no scarring* with the score is **+0.301, the same sign**; *no consolidation* correlates at **−0.534**, more negative than *consolidation*, because 27.6% of vocabulary sentences containing "consolidation" are negated normal-report language and the anchor sits in the normality region of the embedding (*no acute cardiopulmonary process*: −0.601). Every association reported from this instrument is therefore a *topic-neighbourhood* association, not evidence of reliance on a finding, and we withdraw the mechanistic readings an earlier draft had built on them. A re-labelled head (lever 3) moved *scarring* by −0.0344 and *no scarring* by −0.0343. The method's limit is a method limitation and is reported as one.

---

## 7. Exposure, and why no superiority claim is available

Every Vietnam number in this paper is on an exposed cohort. The tuning half was used for selection by design. The reporting half was intended as read-once and was not: a provenance audit found six scoring events across five runs, four of which influenced a decision, including the characterisation of the previous-TB effect on the pooled 2,708. Levers 3, 6 and 9 were each identified on the tuning half, so their combination inherits that exposure; the fused 0.7941 is a best case designed on the cohort, not an estimate of it. TBX11K's split is provisional and ships no patient identifier, so no fold is patient-grouped; 23 of 899 calibration images have a cross-split near-duplicate (effect on the primary endpoint: 0.0003). HIV status was unavailable in every cohort. The localisation result is on 49 development images.

The defensible statement is: *on an exposed screening cohort, a frozen-encoder regional head with clinical-context fusion achieved a numerically higher AUROC point estimate than the commercial comparator (+0.017 [−0.019, +0.053]); the paired interval includes zero, and no formal superiority, equivalence or non-inferiority claim is supported; without the scanner term the point estimate is 0.7803 against 0.7780.* Confidence intervals are participant-level (§2). Turning that into a comparison that survives outside this environment requires a cohort nobody here has touched, the frozen candidate and the comparator scored on the same patients, previous TB as a pre-specified subgroup, and a protocol fixed before the first score.

---

## 8. What we propose

Four protocol additions, none requiring new annotation, that made every distinction in this paper visible:

1. **A region-ablation baseline** (mask the anatomy, re-run the head) evaluated *externally as well as internally*, to separate a dataset-specific shortcut from a genuine signal.
2. **A programmatic screening cohort with a commercial comparator on the same patients.** It converted "0.72 seems low" into "0.72 against 0.79 on identical images". Later, the same comparison produced a numerically higher candidate point estimate (0.794 against 0.778) whose paired confidence interval included zero, with the difference substantially associated with the scanner indicator fields.
3. **A primary split the levers were not chosen on, and a reproduction gate.** Three of our eleven interventions would have been adopted on the tuning half alone; none met its bar on the primary split.
4. **A negation probe before any text-anchored concept score is interpreted.** Cosine between *X* and *no X* above ≈ 0.75 means the instrument cannot say what it appears to say.

---

## 9. Conclusion

A frozen radiology foundation model with a regional head of 769 trainable parameters per seed reaches 0.94 on a benchmark and 0.72 on a screening cohort; a commercial reader reaches 0.79 on the same patients. Eleven pre-registered interventions found three levers that move the screening number and none that moves the benchmark, and the three overlap. With a twelve-field screening questionnaire the best configuration reaches 0.794, numerically above the comparator with a paired interval that includes zero — no superiority, equivalence or non-inferiority claim is supported — and the margin above the comparator is a scanner-associated adjustment that unlabelled centering recovered under the nominal sampling condition tested and lost under population shift. The system's measured properties are a low-memory inference footprint on a consumer laptop GPU, exact attribution with a lesion-localisation result, and a clinical layer from which the comparator's score did not benefit in this experiment; its measured disadvantage is a scanner-associated score offset that the comparator did not show here. We describe it as a scanner-aware screening research system with an experimentally investigated label-free calibration mechanism — not a self-calibrating one, and not a better one — and we think the honest version is the publishable one.

---

## References

[1] Y. Liu, Y.-H. Wu, S.-C. Zhang, L. Liu, M. Wu, M.-M. Cheng, "Revisiting computer-aided tuberculosis diagnosis," *IEEE Trans. Pattern Anal. Mach. Intell.* 46(4):2316–2332, 2024, doi:10.1109/TPAMI.2023.3330825 (TBX11K; SymFormer).
[2] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, G. Thoma, "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases," *Quant. Imaging Med. Surg.* 4(6):475–477, 2014, doi:10.3978/j.issn.2223-4292.2014.11.20 (Shenzhen, Montgomery).
[3] A. J. Codlin, L. N. Q. Vo, T. P. Dao, R. J. Forse, H. T. M. Dang, L. H. Nguyen, H. B. Nguyen, L. V. Dinh, K. Sidney Annerstedt, J. Lundin, K. Lönnroth, "Comparison of different Lunit INSIGHT CXR software versions when reading chest radiographs for tuberculosis," *PLOS Digit. Health* 4(4):e0000813, 2025, doi:10.1371/journal.pdig.0000813; data: Dryad doi:10.5061/dryad.5tb2rbpcv.
[4] A. A. Rege, A. M. Dukre, S. S. Shah, I. Razzak, "RADAR: Acquisition-adversarial attention pooling over a frozen chest-radiograph foundation model for tuberculosis screening," in *MICCAI 2026 Satellite Events — TREAT-MMTB* (Springer proceedings; camera-ready, paper TREAT-MMTB_012); code: github.com/adinathdukre/RADAR-TB. Cited for its architecture; no numerical comparison is made.
[5] Z. Yang, X. Xu, J. Zhang, G. Wang, M. K. Kalra, P. Yan, "Chest X-ray foundation model with global and local representations integration," *IEEE Trans. Med. Imaging* 44(12):4787–4799, 2025, doi:10.1109/TMI.2025.3581907 (CheXFound; GLoRI).
[6] M. Kagujje, A. D. Kerkhoff, M. Nteeni, I. Dunn, K. Mateyo, M. Muyoyeta, "The performance of computer-aided detection digital chest X-ray reading technologies for triage of active tuberculosis among persons with a history of previous tuberculosis," *Clin. Infect. Dis.* 76(3):e894–e901, 2023, doi:10.1093/cid/ciac679.
[7] Z. Z. Qin, M. Van der Walt, S. Moyo, F. Ismail, P. Maribe, C. M. Denkinger, S. Zaidi, R. Barrett, L. Mvusi, N. Mkhondo, K. Zuma, *et al.*, "Computer-aided detection of tuberculosis from chest radiographs in a tuberculosis prevalence survey in South Africa: external validation and modelled impacts of commercially available artificial intelligence software," *Lancet Digit. Health* 6(9):e605–e613, 2024, doi:10.1016/S2589-7500(24)00118-3.
[8] F. Pérez-García, H. Sharma, S. Bond-Taylor, K. Bouzid, V. Salvatelli, M. Ilse, S. Bannur, D. C. Castro, A. Schwaighofer, M. P. Lungren, M. T. Wetscherek, N. Codella, S. L. Hyland, J. Alvarez-Valle, O. Oktay, "Exploring scalable medical image encoders beyond text supervision," *Nat. Mach. Intell.* 7:119–130, 2025, doi:10.1038/s42256-024-00965-w (RAD-DINO; `microsoft/rad-dino`, `microsoft/rad-dino-maira-2`).
[9] T. Han, R. Wu, Y. Tian, F. Khader, L. C. Adams, K. K. Bressem, C. Davatzikos, J. N. Kather, L. Shen, D. A. Mankoff, E. Mortani Barbosa Jr., D. Truhn, "CLEAR: an auditable foundation model for radiology grounded in clinical concepts," *Nat. Biomed. Eng.*, 2026, doi:10.1038/s41551-026-01741-4; code and vocabulary: github.com/peterhan91/CLEAR (revision pinned in the repository).
[10] Stanford AIMI, *XraySigLIP (ViT-B/16, SigLIP, 512 px)*, model card, Hugging Face, model id `StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli`, revision `e2ad64eee3c2a2ec746f4c0934b018c718917714`, accessed 2026-11-01. The card cites Z. Chen *et al.*, "CheXagent: Towards a foundation model for chest X-ray interpretation," arXiv:2401.12208, 2024, which is not itself an XraySigLIP paper.
[11] M. Akhtar, M. Tanveer, M. Arshad, "Beyond benchmark scores: Auditing medical vision-language models for chest X-ray tuberculosis screening," arXiv:2609.21763, 2026 (zero-shot TBX11K reference points: MedSigLIP 0.800, CheXficient 0.795).

*Citation-to-claim audit (2026-11-01): [1] benchmark and 99.1% AUROC claim; [2] external sets; [3] cohort, comparator scores and the previous-TB finding on this cohort; [4] and [5] prior art for frozen-encoder patch aggregation, not compared numerically; [6] and [7] prior reports of reduced CAD accuracy in previous TB (Kagujje: specificity 24.0% vs 60.3% at 90% sensitivity for CAD4TB; Qin: subgroup decline across products); [8] encoders; [9] the probing instrument of §6; [10] the third encoder of §5; [11] zero-shot reference points (§1). Author list for [8] to be abbreviated per venue style before submission.*

---

## Notes for the team (not part of the submission)

- **Every TBX11K number is on the provisional split.** Regenerate when C01 is frozen; the run scripts take minutes.
- **Nothing in §5 is validation.** §7 says so; keep it saying so.
- **The earlier draft's §3 (border shortcut), §6 (OOD gate) and §7 (proposals) are condensed here** and remain valid; its concept-interpretation passages are superseded by §6.
- **Reproduction:** every number traces to a run directory under `artifacts/` with a hashed protocol, tables and tests; the project record `artifacts/P2_PROJECT_RECORD_20261101.md` indexes them (R1–R25).
- **Candidate v3** (`artifacts/p2/bundle_v3_candidate`) is the frozen system for any future head-to-head; the served default remains v2.
- **Authorship and scope** to be agreed; §3.1 and the benchmark-duplicate findings touch P1's workstream.
