# The Vietnam cohort breaks two of our claims and confirms the third

**Owner:** Person 2 · **Date:** 2026-10-19 · **Status:** exploratory (sealed run still
required) · **Cohort:** 2,708 participants, 280 Xpert Ultra-positive, Ho Chi Minh City
community active case-finding, CC0

This is the first time the system has met a **consecutive screening cohort** rather than a
curated case-control set. The result is the most important in the project and two thirds of
it is negative.

## 1. The system does not work here: AUC 0.6151

| what the model can see | AUC | 95% CI |
|---|---|---|
| full image | **0.6151** | [0.5853, 0.6452] |
| lungs only | 0.6430 | [0.6128, 0.6728] |
| border only | 0.5474 | [0.5140, 0.5811] |
| **Lunit INSIGHT 3.9.0.1 (commercial)** | **0.7827** | [0.7538, 0.8105] |
| Lunit INSIGHT 3.1.0.0 (commercial) | 0.7633 | [0.7357, 0.7893] |

**A certified commercial product reaches 0.7827 on exactly these patients and these images.
We reach 0.6151.** The task is demonstrably doable at 0.78 on this cohort; our system is
simply not good enough for it. That gap is not explainable by the data being hard.

Against the four sites measured before:

| site | type | full | lungs-only | border-only |
|---|---|---|---|---|
| TBX11K (home) | curated benchmark | 0.9949 | 0.9889 | 0.9094 |
| Shenzhen, China | curated case-control | 0.9382 | 0.9284 | 0.5209 |
| Montgomery, USA | curated case-control | 0.8183 | 0.8554 | 0.6082 |
| DA+DB, India | curated case-control | 0.7932 | 0.7791 | 0.6240 |
| **Vietnam** | **consecutive screening cohort** | **0.6151** | 0.6430 | 0.5474 |

At the TBX11K-fitted threshold, installation without recalibration gives **sensitivity
0.8071 at specificity 0.3752** — it would refer nearly two-thirds of healthy attendees. With
local recalibration, **sensitivity at 70% specificity is 0.3893**: it would miss three TB
cases in five.

Per scanner: 0.6515 (JPI), 0.7133 (DRTECH), 0.5960 (Konica), against Lunit's 0.7809 /
0.8095 / 0.7422 on the same subgroups. We are worse on every machine.

## 2. Why: the model learned "abnormal", and in a screening queue everyone is abnormal

The diagnosis is visible in the score distribution:

| | median score, TB | median score, non-TB |
|---|---|---|
| ours | 0.9530 | **0.8856** |
| Lunit | 0.8890 | **0.5680** |

**Our model calls almost everything tuberculosis.** It separates the two groups barely at
all, while Lunit holds its negatives down at 0.57. Spearman correlation between the two
scores is 0.512 — we are detecting something related, much more weakly.

This follows directly from a result we already had and did not join up.
`p2_control_group.md` found that TB versus *healthy* is AUC 1.0000 while TB versus
*sick-but-not-TB* is 0.9861 — on TBX11K, where "sick" is a curated category. A consecutive
screening cohort is overwhelmingly the second case and then some: its Xpert-negatives include
old healed tuberculosis, COPD and other chronic disease, and its positives are
**screen-detected** — often early, minimally symptomatic, with subtle radiographic change —
rather than the florid, lesion-annotated cases TBX11K is built from.

The model was trained to recognise images that look abnormal. In a screening queue, that
describes most of the queue.

## 3. The OOD gate does not catch it — and this falsifies our own claim

`p2_external_validation.md` claimed that site distance is measurable in advance from
unlabelled images and **predicts transfer**, with the ordering TBX11K 0.364 < Shenzhen 0.533
< CheXpert 0.779 matching the ordering of the AUCs. Vietnam breaks it:

| site | median OOD score | gate rejection | full-image AUC |
|---|---|---|---|
| TBX11K | 0.3637 | 0.2% | 0.9949 |
| Shenzhen | 0.5331 | 1.2% | 0.9382 |
| **Vietnam** | **0.5312** | **0.8%** | **0.6151** |

**Vietnam sits at the same distance as Shenzhen and performs 0.32 AUC worse.** The gate
passes 99.2% of this cohort as in-distribution, and the classifier is then wrong about them.

So the earlier claim was fitted to three points and is **withdrawn**. The honest version:

> The OOD score detects whether images *look like* the training distribution. It does not
> detect whether the *task* is the same. A screening cohort can be radiographically
> unremarkable to a kNN gate while being a completely different classification problem,
> because what changed is the composition of the negative class, not the appearance of the
> images.

This is the more useful statement, and it is a real limitation of distance-based OOD gating
generally, not just of our implementation. A gate that cannot see a prevalence or
case-mix shift will wave through exactly the deployment that is about to fail.

## 4. What survives: the shortcut finding, strengthened

Border-only scores **0.5474 [0.5140, 0.5811]** against its TBX11K home value of 0.9094.

This is the cleanest confirmation available, because of a property unique to this cohort:
**the scanner does not predict the outcome.** Three radiography systems, TB rates of 10.8%,
11.3% and 8.8%, and `rad_system` as a predictor gives AUC 0.4722 — chance. Acquisition
*cannot* encode the label here by construction. The border signal duly disappears.

Four sites now: 0.5209, 0.6082, 0.6240, 0.5474, against 0.9094 at home. The shortcut is a
TBX11K-specific artifact and it does not travel. That claim is now firm.

**Lungs-only beats the full image again** (0.6430 vs 0.6151), as at Montgomery. At every
site that is genuinely distant, discarding everything outside the lung field helps. This is
now three sites of evidence for reopening D09 in favour of the lung crop.

## 5. What this means for the project

**The headline must change.** Leading with 0.9949, or even with Shenzhen's 0.9382, would be
indefensible now. The honest summary is:

> On curated case-control data the system reaches 0.94–0.99 AUC. On a consecutive screening
> cohort with a microbiological reference standard it reaches 0.6151, against 0.7827 for a
> certified commercial product on the same patients. The benchmark result does not predict
> the deployment result, and our out-of-distribution gate does not detect the difference.

That is a better paper than the one we were writing. The existing draft argues that internal
numbers are contaminated and external ones are clean; this shows **"external" is not one
thing** — curated external validation is much closer to the benchmark than to deployment, and
the field's standard external sets (Shenzhen, Montgomery) are the easy kind.

**It also gives us a target.** Lunit's 0.7827 is a concrete, same-patient demonstration that
0.78 is attainable here. Candidate explanations for our gap, none yet tested:

1. **Training distribution.** TBX11K's TB cases are lesion-annotated and therefore
   radiographically evident. Screen-detected TB is subtler. A model trained only on visible
   disease may be unable to see early disease at all.
2. **Negative-class composition.** 5,000 of TBX11K's 11,200 are "sick non-TB", but curated
   sick, not a screening population.
3. **Frozen backbone.** A linear probe on frozen features may lack the capacity for a harder
   discrimination; the P2-17 fine-tuning extension becomes much more interesting now.
4. **Intensity shift.** Per-scanner means 119/125/164 against TBX11K's ~128. AUC is
   rank-based so a monotone shift should not hurt it, which makes this the least likely
   explanation — but it is testable.

## Caveats

- Exploratory. Heads trained on TBX11K and applied unchanged; `internal_test` untouched. The
  reportable figure must come from a sealed run.
- HIV status was dropped by the depositors, and HIV strongly alters tuberculosis
  presentation. In a Vietnamese screening cohort HIV prevalence is likely low, but we cannot
  verify it. The Zambia cohort (35.5% HIV-positive) would let us measure this directly.
- One segmentation sanity failure in 2,708 (0.04%).
- DICOMs were normalised through the VOI LUT with a 0.5/99.5 percentile clip, deliberately
  *not* histogram-matched to TBX11K, since tuning preprocessing to a test set is a soft form
  of fitting to it.

Evidence: `artifacts/p2/external/vietnam_eval.json`, via `scripts/p2_27_vietnam_ingest.py`
and `scripts/p2_28_vietnam_eval.py`.
