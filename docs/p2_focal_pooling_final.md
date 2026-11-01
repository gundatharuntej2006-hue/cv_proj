# Focal pooling: the complete result

**Owner:** Person 2 · **Date:** 2026-10-20 · **Supersedes:** `p2_focal_pooling.md` (the
preliminary 6×6 run) · **Primary result read once and locked**

## Headline

Replacing RAD-DINO's globally pooled output with **max-pooling over a 12×12 grid of region
scores** reaches **0.7242 [0.6843, 0.7618]** on the reserved Vietnam **REPORT** half
(n = 1,356, 141 TB), using the same backbone, the same training data, and no fine-tuning.
The earlier configurations on this cohort scored **0.6151** and **0.6743**, but both were
measured on the **full 2,708-participant cohort**, so **0.6151 → 0.7242 is not a
like-for-like before/after** and must not be quoted as one improvement. The table below
states each figure with its own evaluation set.

> **Cohort provenance (added 2026-10-21).** The screening cohort is the Dryad test library
> [10.5061/dryad.5tb2rbpcv](https://doi.org/10.5061/dryad.5tb2rbpcv), identified on the
> evidence in `docs/p2_vietnam_result.md` as the dataset of Codlin et al., *PLOS Digital
> Health* 2025;4(4):e0000813
> ([doi:10.1371/journal.pdig.0000813](https://doi.org/10.1371/journal.pdig.0000813)).
> Identification is by **statistical reproduction** — matching title, author list, and six
> published AUROCs to two decimal places — **not by patient-ID matching**; participant-level
> identity was not verified. Recruitment was **risk-based programmatic active case
> finding** (contacts, adults ≥55 and symptomatic people mobilised ahead of 127 mobile
> screening events), so it is not an unselected consecutive cohort, though it remains a
> screening population rather than a curated case-control set. The Lunit comparator figure
> below is that paper's published comparator, not a benchmark assembled here.

> **These rows are not a like-for-like before/after.** Two things change down the table,
> not one: the configuration *and* the evaluation set. The first two rows are the **full
> 2,708-participant cohort**; the last two are the **reserved REPORT half (n = 1,356, 141
> TB)**. A global-pooling score on the REPORT half alone was never computed, so the
> progression cannot be put on one denominator, and **the 0.6151 → 0.7242 difference must
> not be quoted as a single improvement.** Each row is stated with its own evaluation set
> below.

| configuration | evaluation set | n | TB | AUROC | source |
|---|---|---|---|---|---|
| full image, all negatives, global pooling | full cohort | 2,708 | 280 | 0.6151 | `vietnam_eval.json`, variant `full`, scope `all` |
| lungs-only masking, healthy-only negatives, global pooling | full cohort | 2,708 | 280 | **0.6743** [0.6443, 0.7039] | `hard_negatives.json`, `lungs_only` × `healthy_only`, 5 seeds |
| lungs-only, healthy-only, **focal pooling (12×12, max)** | **REPORT half** | **1,356** | **141** | **0.7242** [0.6843, 0.7618] | `mil_final.json`, Vietnam REPORT |
| *commercial comparator, Lunit INSIGHT 3.9.0.1* | *REPORT half* | *1,356* | *141* | *0.7862* | *`mil_final.json`* |
| *commercial comparator, Lunit INSIGHT 3.9.0.1* | *full cohort* | *2,708* | *280* | *0.7827* | *`vietnam_eval.json`, `lunit_3901`* |

**What can be said on one denominator.** On the **full cohort**, the configuration change
from the first row to the second is **+0.059** (0.6151 → 0.6743), and the gap to the
commercial comparator there is **0.168** (0.7827 − 0.6151). On the **REPORT half**, the
gap to the comparator is **0.062** (0.7862 − 0.7242). **These two gaps are measured on
different populations and the narrowing from 0.168 to 0.062 therefore conflates the
configuration change with the change of evaluation set.** The focal-pooling contribution
on its own is established not by this table but by the pre-registered TUNE/REPORT protocol
and the matched `mean` control in §"The sweep" below.

> **This architecture is not novel.** Patch-level aggregation over a frozen chest-radiograph
> foundation model for tuberculosis is established prior art — see §"Relation to prior work"
> below. What this document reports is a *controlled measurement* of the aggregation choice
> under one pipeline, not a new method.

## Protocol, because the number depends on it

Searching aggregators against the test cohort would overfit it — the failure this project
exists to document. So the cohort was split **before any tuning**, stratified by outcome and
scanner, seeded, with the reporting half hashed:

- **TUNE** 1,352 participants, 139 TB — every design decision
- **REPORT** 1,356 participants, 141 TB — read exactly once, at the end

The scoring script was committed **before the sweep results were examined**, so the
reporting rule could not be retrofitted to the winner.

**Selection optimism was −0.0005**: TUNE 0.7237 against REPORT 0.7242. A configuration chosen
on one half transferred perfectly to the other, which is the strongest available evidence
that the gain is real rather than tuning noise.

## The sweep

Eight region grids, five aggregators, three seeds, selected on TUNE only. `mean` is
mathematically equivalent to global pooling and is the control.

| grid | mean (control) | max | top3 | lse |
|---|---|---|---|---|
| 1×1 | 0.6381 | — | — | — |
| 2×2 | 0.6395 | 0.6808 | 0.6561 | 0.6779 |
| 3×3 | 0.6422 | 0.6500 | 0.6589 | 0.6559 |
| 4×4 | 0.6419 | 0.6649 | 0.6674 | 0.6721 |
| 6×6 | 0.6390 | 0.6888 | 0.6858 | 0.6822 |
| 9×9 | 0.6368 | 0.7205 | 0.7110 | 0.7142 |
| **12×12** | 0.6364 | **0.7239** | 0.7189 | 0.7229 |
| 18×18 | 0.6349 | 0.7186 | 0.7224 | 0.7257 |

Three things to read from it.

**The control is flat.** `mean` sits at 0.636–0.642 at every grid. Pooling more finely and
then averaging *is* global pooling, so it must be flat, and it is. That rules out the
alternative explanation that finer grids simply give the head more parameters to fit with.

**Performance saturates at roughly lesion scale.** The curve climbs steeply to 9×9, then
flattens: 6×6 0.6888, 9×9 0.7205, 12×12 0.7239, 18×18 0.7186. Each region at 12×12 is about
43 px of the 518 px input, which is the scale of a tuberculosis lesion. Resolving finer than
the thing you are looking for does not help.

**18×18 + lse is nominally highest at 0.7257, and was not selected.** It exceeds 12×12 + max
by 0.0018, far below noise. Switching the frozen configuration after seeing a new number, to
capture 0.0018, is precisely the behaviour the pre-commitment exists to prevent. Recorded in
`mil_sweep.json` under `grid18_addendum` and deliberately not acted on.

## What did not work

Attention-based MIL (Ilse et al. gated attention) and a two-layer scorer, both selected on
TUNE after the primary result and therefore **exploratory**:

| method | TBX11K dev | TUNE |
|---|---|---|
| max-pooling | 0.9369 | **0.7239** |
| gated attention | 0.5950 | 0.6072 |
| gated attention + MLP | **0.9495** | 0.6416 |

Learned attention is substantially worse than taking the maximum. The plausible reason is
that attention helps when many weak instances must be combined, whereas tuberculosis is
typically one dominant focus — exactly the case max-pooling is built for.

Note the second row: **attention+MLP achieves the best internal fit of anything built in this
project (0.9495) and transfers worst.**

## Relation to prior work

**The regional-pooling architecture is established, not novel.** Three sources, with
verification status stated:

| source | status | overlap with this work |
|---|---|---|
| **RADAR**, *Acquisition-Adversarial Attention Pooling over a Frozen Chest-Radiograph Foundation Model for Tuberculosis Screening*, MICCAI 2026 satellite — [papers.miccai.org/miccai-2026-sat/TREAT-MMTB_012.html](https://papers.miccai.org/miccai-2026-sat/TREAT-MMTB_012.html) | **retrieved and read** | Closest TB-specific prior art. Frozen CXR foundation model, multi-depth patch representations, a learned TB query cross-attending over patches, modality- and conditional-adversarial objectives, style randomisation with prediction consistency, seed/view ensembling, and a higher-resolution read-out explicitly targeting small focal abnormalities. |
| **CheXFound / GLoRI**, *Chest X-Ray Foundation Model With Global and Local Representations Integration*, IEEE TMI 44(12), 2025 — [doi.org/10.1109/TMI.2025.3581907](https://doi.org/10.1109/TMI.2025.3581907) | **cited from the task brief; full text not retrieved in this environment** | Establishes global + local feature integration as a studied adaptation strategy for CXR foundation models. Its primary task setting differs from this TB experiment. Treated as unverified. |
| **RAD-DINO model card** — [huggingface.co/microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino) | **retrieved and read** | Backbone provenance: trained on MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX (882,775 images). The released checkpoint was selected by linear probing on validation sets the card does not name. |

**Where this work differs from RADAR**, stated without implying superiority:

- RADAR *learns* attention over patches; we take the maximum. We measured learned gated
  attention and found it **worse** here (0.6416 against 0.7239 on our tuning half) — a
  finding about a low-data regime, not a refutation of their design.
- RADAR actively *mitigates* acquisition shortcuts with adversarial objectives and style
  randomisation. **We only diagnose and quantify them.** On this axis we are behind the
  prior art, not ahead of it.
- RADAR uses multi-depth patch features and higher-resolution read-out; we use a single
  depth at 12×12.
- Our evaluation adds a screening cohort — risk-based programmatic active case finding,
  not unselected consecutive enrolment — with an Xpert reference standard and a commercial
  comparator on identical patients, which the RADAR page does not report. That is an
  evaluation difference, not an architectural one. Note that the comparator and the
  previous-TB subgroup analysis on this cohort are themselves published (Codlin et al.
  2025), so this is not an independent evaluation asset of ours.

**Accordingly, the defensible contribution here is the controlled measurement** — matched
training data, matched seeds, an internal control that is mathematically equivalent to
global pooling, and pre-registered selection — not the aggregator itself.

## The pattern this makes six of

Every intervention that improved external transfer made the TBX11K fit worse, and every
intervention that improved the TBX11K fit made transfer worse:

| intervention | TBX11K | external |
|---|---|---|
| border-only features | 0.9094 | collapses to chance |
| adding sick negatives | better | worse |
| more training data | flat / better | flat / worse |
| attention + MLP capacity | 0.9495, best | 0.6416, worst |
| focal pooling | 0.9889 → 0.9368 | 0.6743 (full cohort) → 0.7242 (REPORT half) — different denominators, see the Headline note |

On this benchmark, **fitting it better makes you transfer worse**. That is now the clearest
regularity in the project and arguably its most useful single sentence: TBX11K development
performance is not merely an unreliable guide to deployment, it is an actively misleading
one.

## What it means clinically

AUC is abstract. On the held-aside half — 1,356 people, 141 with tuberculosis:

| if we accept wrongly flagging | we detect | commercial system detects |
|---|---|---|
| 30% of healthy people | 59% | 65% |
| 20% | 44% | 60% |
| 10% | 24% | 47% |

Per 1,000 screened at 70% specificity: **catch 61 cases, miss 43, wrongly refer 269 healthy
people.**

The WHO target product profile for a TB triage test is **90% sensitivity at 70%
specificity**. We reach 59%. The commercial comparator reaches 65%. **Both fail the WHO bar
on this cohort** — worth stating, because it establishes that this is a hard population
rather than a uniquely bad model.

## Limitations

- Exploratory with respect to the final split: provisional C01, `internal_test` untouched.
- One screening cohort. The curated sites have not been re-scored under focal pooling, so
  whether it helps or harms there is unknown.
- Grids beyond 18×18 were not reachable on this hardware (the full 37×37 patch grid needs
  about 28 GB during array construction), but the curve had already flattened by 9×9, so this
  is unlikely to be binding.
- Max-pooling is sensitive to a single outlying region, a plausible failure mode on images
  with artefacts; `top3` is the safer choice and is 0.005 behind.
- Changing the aggregator changes the score scale, so calibration and the triage thresholds
  would need refitting. Only AUCs are compared here.

Evidence: `artifacts/p2/regional/mil_sweep.json`, `mil_final.json`, `attention_mil.json`;
scripts `p2_31`, `p2_32`, `p2_33`, `p2_34`.
