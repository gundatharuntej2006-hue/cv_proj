# Slide deck v2 — 15-minute talk, P2 component

*Owner: Atul (P2-16). Content draft; formatting decided at assembly. Every number traces to the project record (`artifacts/P2_PROJECT_RECORD_20261101.md`); the slide says which run. Supersedes `p2_slides_outline.md` (pre-R15).*

**Structure.** Three minutes on the question and the system, six on what was found, four on what was tried and what it means, two on what is next. Fifteen slides, one idea each.

**Speaker's rules.**
1. Never show 0.9368 without 0.7242 on the same slide.
2. Never say "better than Lunit". Say "a numerically higher point estimate with an interval that includes zero".
3. Every Vietnam number is on an exposed cohort — say it once, early (slide 6), and point to slide 13 when asked.
4. The failures are the talk. Rejected experiments get the same font size as the successful ones.

---

## 1 — Title

**Explainable TB Screening from Chest X-Rays with Lesion Localisation for Triage**
BAI506 Mini Project · Department of AI & ML, BMSIT&M · 2026–27
Tharun · Atul Dhull · Ritika · Kunal

*P2 component: classifier, evaluation, calibration, serving.*

---

## 2 — The question

- WHO recommends chest-X-ray CAD for TB screening and triage in people ≥ 15 (2021), target 90 % sensitivity at 70 % specificity
- Published methods: **99.1 % AUROC** on TBX11K, the largest public benchmark
- Commercial systems exist, are closed, and are validated mostly by their makers
- **Our question:** what survives when a public model is evaluated the way a screening programme would use it — on a real screening cohort, with a microbiological reference, against a commercial reader on the same patients?

*Say: we did not set out to lose to Lunit. We set out to measure honestly, and then to close whatever gap appeared.*

---

## 3 — What we built (the pipeline)

Diagram, left to right:

`X-ray → standardise 512² → quality gate → lung mask (PSPNet) → frozen RAD-DINO ViT-B/14 → 12×12 region vectors → Linear(768→1) per region → MAX over 144 regions → image logit → operating point → optional questionnaire fusion`

- **Frozen encoder (86.6 M params), 769 trainable per seed × 5 seeds, +13 for fusion**
- The image logit *is* the maximum region logit → **the arg-max region is the decision, exactly** (no post-hoc heatmap)
- Served: `/v2/batch` ranks a batch; `/v2/overlay` draws the decision cell; triage screen at `/v2/ui`
- **0.13 s per image** median on a 6 GB laptop GPU (segmenter on GPU; R-eng 2026-11-01)

*Say: nothing here is architecturally new — frozen-encoder patch heads exist (RADAR, CheXFound). The contribution is what we measured with it.*

---

## 4 — Data: a ladder, not a benchmark

| rung | cohort | n | what it is |
|---|---|---|---|
| benchmark | TBX11K | 8,400 core (800 TB), provisional split 5,941 / 659 / 899 / 901 | curated, lesion boxes, one image per person, **no patient id** |
| curated external | Shenzhen, Montgomery, DA/DB | 662 / 138 / 176 | case-control, same-source controls |
| **screening** | **Vietnam** (Codlin 2025, Dryad) | **2,708**, 280 Xpert+, 36 % previous TB, 3 scanners, questionnaire, **Lunit 3.9.0.1 scores** | programmatic risk-based active case finding (not consecutive) |

- Vietnam split once before any tuning: **TUNE 1,352 / REPORT 1,356**
- The comparator's published AUROCs reproduce from our copy to two decimals

---

## 5 — The benchmark number is partly a border (R1–R4)

- Global probe: **0.9949** on TBX11K development
- …and **0.9094 from the outer 12 % of the image alone** — shoulders, collimation, no lung
- Dilate the erased lungs by 48 px: −0.0039. Flip horizontally: +0.0001
- The shortcut **does not transfer**: 0.52–0.62 at the four external sites; lungs-only tracks the full image everywhere
- → production model is lung-masked; **TB-vs-healthy on TBX11K is AUROC 1.0000** — the metric is saturated

*Show the border-only image next to the full image.*

---

## 6 — The ladder (the central slide)

| rung | regional 12×12 max | comparator |
|---|---|---|
| TBX11K development | **0.9368** | — |
| Shenzhen / Montgomery | 0.9690 / 0.9690 | — |
| DA / DB | 0.9755 / 0.9497 | — |
| **Vietnam REPORT** | **0.7242 [0.6843, 0.7618]** | **0.7862** |

- Same frozen model, same head. 0.21 AUROC between benchmark and clinic
- Selection optimism TUNE − REPORT = −0.0005 (the sweep did not overfit the tuning half)
- **Exposure, stated now:** TUNE was used for every selection; REPORT was scored six times across the project (four influenced decisions). Neither is a clean test set. Slide 13 says what that means.

*Say: the screening cohort is where the task is. A curated external set is not that test.*

---

## 7 — Where the loss is: previous tuberculosis (R5–R7)

- Without previous TB **0.7720** → with **0.6392** (degradation +0.1328 [0.0718, 0.1951]) — construction used by the published literature
- Comparator: 0.8170 → 0.7192
- Difference-in-differences **+0.0481 [−0.0170, +0.1113]** — includes zero: **we cannot claim to degrade more than the comparator**
- This effect was **published by the dataset's own authors** and by Kagujje 2023 and Qin 2024 — we reproduce it, we did not discover it
- Scanner *appearance* is not the cause: removing the scanner-predictive subspace changes AUROC by −0.0002

---

## 8 — Eleven pre-registered interventions (R3–R25)

Every row: protocol hashed before fitting, reproduction gate, primary split = TBX11K calibration, bar = +0.02 with interval excluding zero.

| intervention | calibration (primary) | Vietnam TUNE | verdict |
|---|---|---|---|
| active-TB label (latent → negative) | −0.0037 | **+0.0203 [+0.0127, +0.0279]** | external only |
| clinical-context fusion, 12 fields | — | **+0.058 [+0.030, +0.085]** | external only |
| RAD-DINO-MAIRA-2 encoder | −0.0004 | **+0.0188 [+0.0015, +0.0364]** | external only |
| three training sources | +0.0033 [+0.0006, +0.0064] | +0.0036 | real, sub-threshold |
| lesion-box supervision | +0.0019 | −0.0108 | null |
| encoder ensemble (3) | +0.0006 | +0.011 | null |
| head refit vs threshold-only | 0 of 30 help | — | null |
| gated attention; scanner subspace | — | −0.082; −0.0002 | worse; null |
| all three levers stacked | −0.0010 | +0.0286 (predicted +0.047) | **not additive** |

*Say: three things help on the clinic and none on the benchmark — and they overlap. More capacity transferred worse every time (2,304-d head: best dev 0.9505, worst transfer). That is why the backbone was never fine-tuned.*

---

## 9 — Fusion: what the questionnaire actually buys (R16, R22, R23)

| score | raw | fused (12 fields) |
|---|---|---|
| production head | 0.7237 | 0.7809 |
| MAIRA-2 + active head | 0.7585 | **0.7941** |
| Lunit | 0.7780 | 0.7778 (+0.001) |

- Fused candidate vs Lunit raw: **+0.0170 [−0.0193, +0.0527]** — point above, interval includes zero
- Leave-one-field-out: only the **scanner identifier** carries the gain (−0.0138; −0.043 in previous TB). `past_tb` itself: +0.004
- Questionnaire only, no scanner term: **0.7803 vs 0.7780** — within noise
- **The margin above the comparator is a per-scanner intercept.** The comparator did not benefit from the same fields in this experiment — that does not make either system scanner-invariant

---

## 10 — Removing the scanner offset without labels (R25)

- Mean-centre each scanner's scores on **25 unlabelled** local images → **0.7795**; at 100 images, paired **+0.0281 [+0.0074, +0.0486]** over raw — same as the supervised term
- Referral burden across scanners 31 / 24 / **50 %** → 40 / 31 / 32 %
- **Pre-registered failure:** a calibration batch of previous-TB patients only drops that scanner's sensitivity 0.677 → 0.555 (bound 0.10) → **NOT USABLE AS IS**
- Supervised term fails worse under prevalence shift (sens 0.18 on negatives-only batch)
- Shipped as a **fail-closed proposal endpoint**: refuses below 25 images, without a declared site reference, or on a composition shift; fallback is always the frozen model. Experimental, not validated.

*Say: this is the slide examiners should push on. The failure is the result.*

---

## 11 — Explainability that is exact (R17)

- Image logit = max over 144 region logits → the decision cell is attribution **by construction**, at 12×12 (≈ 43 px), not a lesion boundary
- 49 development images with active-lesion boxes: decision cell overlaps a box in **95.9 % [89.8, 100]** (top-3: 100 %) vs 36.7 % random lung cell, 18.4 % random cell, 8.2 % centre
- Box-supervising the head: 0.939 — no improvement; the detector stays a separate component
- Served overlay (show `docs/p2_overlay_example_tb0037.png`): green cell on the right apex, logit 4.34

---

## 12 — A negative result on concept probing (R9–R14)

- Projected the cohort through a radiology vision–language model (CLEAR); correlated the score with text anchors
- *scarring* ρ = 0.566 with our score — and AUROC 0.494 against Xpert (no TB information); *consolidation* −0.300 (wrong sign)
- Forensic pass: ***no scarring* +0.301, same sign; *no consolidation* −0.534, more negative** — the anchors cannot tell a finding from its negation (27.6 % of consolidation sentences in the vocabulary are negated normal-report language)
- **Every mechanistic reading withdrawn.** Method limitation, reported as one. Rule: probe negation before interpreting any text-anchored concept score

---

## 13 — What the evidence supports, and what it does not

**Supports**
- Near parity with commercial CAD on an exposed screening cohort: +0.017 [−0.019, +0.053] fused; 0.7803 vs 0.7780 without the scanner term
- Exact region attribution with a measured 95.9 % localisation result (exploratory, n = 49)
- 0.13 s/image on a laptop GPU; 3,845 + 13 trainable parameters

**Does not**
- Superiority, equivalence or non-inferiority (interval includes zero; levers chosen on the same cohort)
- Scanner invariance, in either system
- Safe automatic calibration at a new site
- Previous-TB performance better than the comparator's

*Say: a scanner-aware screening research system with an experimentally investigated label-free calibration mechanism. Not self-calibrating. Not better.*

---

## 14 — What is next, and what it will take

- **A cohort nobody here has seen.** Pre-registered protocol, hashed: non-inferiority at margin 0.03, previous TB pre-specified, scanner-stratified, **≥ 400 confirmed cases**; one-shot scorer with a LOCK; data request ready (TB Portals, HCMC group, Bengaluru centres)
- Definitive TBX11K split (P1) — every benchmark number is conditional on it
- Prospective test of the calibration guard
- Detector ↔ decision-cell agreement once P3's detector is final

*Say: twenty-five documented experiments were enough to know the ceiling of this data. The next number has to come from new patients, not new models.*

---

## 15 — Objectives, honestly

| proposed | outcome |
|---|---|
| ≥ 90 % sens at ≥ 70 % spec on a held-out internal split | met on development (not independent); **not met** on the screening cohort (0.62 raw, 0.74 fused) |
| three-class classifier | no difference vs binary (0.9661 both); binary adopted |
| Shenzhen / Montgomery evaluation | 0.9690 / 0.9690 — curated sets overstate screening by 0.2 |
| web tool, < 2 s/image | **met**: ranked batch, overlay, screen; 0.13 s/image |
| detector, challenge submission | P1/P3 component; no submission |

**Thank you.** Record, code, protocols and tests: branch `atul` on the team repository.

---

## Backup slides (only if asked)

- **B1 — OOD gate:** kNN gate passes 99.2 % of the Vietnam cohort at the same distance as Shenzhen; appearance shift ≠ task shift (record §3.2)
- **B2 — R19 multi-source:** +0.0033 [+0.0006, +0.0064] — real, one sixth of the bar; MB1 0.7517 on TUNE, not adopted
- **B3 — Capacity transfers worse:** global heads 0.957–0.995 → 0.67; attention 0.9495 → 0.6416; concat 0.9505 → −0.0137
- **B4 — Power table:** SE ≈ 0.018 at 139 positives; 410 cases for non-inferiority at 0.03; 930 for superiority at +0.02; previous-TB subgroup ≈ 650 cases — exploratory at any realistic size
- **B5 — Dataset defects (for P1):** `tb1199` label; 140 empty `active_tb_label`; 158 cross-split near-duplicate pairs; 23/899 calibration near-dups (effect 0.0003)
