# Concept note: does a model-specific previous-TB discrimination gap reproduce?

*Prepared 2026-10-21 for discussion. Not a funding application. No data access is assumed.*

---

## The problem

Chest-radiograph AI for tuberculosis triage performs well on curated benchmarks and less
well in screening populations. A recognised contributor is **previous tuberculosis**:
treated disease leaves radiographic sequelae that resemble active disease, and people
previously treated for TB are over-represented in exactly the populations where active
case finding is done.

## What is already known

Kagujje et al. (*Clin Infect Dis* 2022;76(3):e894–e901, DOI 10.1093/cid/ciac679)
established this in 1,884 Zambian ACF participants, 452 (24%) with self-reported previous
TB. At thresholds set for ~90% overall sensitivity, **specificity fell from 60.3% to 24.0%
for CAD4TB and from 61.8% to 22.2% for qXR**. Crucially, discrimination behaved
differently between products: the main text reports that **CAD4TB's AUROC was higher in
participants without previous TB, while qXR's was comparable across the two subgroups.**

Qin et al. (*Sci Rep* 2025;15:22540) add that **radiologists are affected too** — every
reader group in their multi-country comparison showed significantly lower specificity in
prior-TB participants. So this is not a property of AI alone.

**The phenomenon is established. We are not proposing to discover it.**

## What we observed

We built a research prototype on a frozen chest-radiograph foundation model (RAD-DINO),
with a lightweight trained read-out. On the publicly released Vietnamese active
case-finding cohort (Dryad, n = 2,708, Xpert Ultra reference, 280 positives), stratifying
by self-reported previous TB — present in 35% of Xpert-negatives:

| | our prototype | Lunit INSIGHT CXR 3.9.0.1, identical images |
|---|---|---|
| vs negatives **without** previous TB | 0.7811 | 0.8170 |
| vs negatives **with** previous TB | **0.6182** | **0.7192** |
| **degradation** | **0.1629 [0.1372, 0.1890]** | 0.0978 [0.0770, 0.1196] |

**Difference in differences: +0.0651 [0.0399, 0.0900]** — our degradation is larger than
the comparator's on the same participants. Intervals come from a stratified bootstrap that
reuses the same positive resample in each replicate, because both subgroups share all 280
positives.

Because AUROC is threshold-free, this gap **cannot be repaired by recalibration or a
subgroup-specific threshold**.

## The honest limitations

- **Exploratory.** The subgroup was chosen after screening five clinical variables; it was
  not prespecified, and the intervals are not selection-corrected.
- **Exposed cohort.** This dataset informed earlier configuration choices in our work.
- **Observational.** Previous TB is self-reported and correlates with age, comorbidity and
  prior care-seeking. We cannot attribute the gap to radiographic sequelae.
- **Not the whole story.** Our AUROC is 0.7811 even *without* previous TB, against the
  comparator's 0.8170 — a separate 0.036 deficit.
- **Possibly not new.** A *Lancet Digital Health* 2024 companion paper to Qin et al. may
  already report per-product prior-TB AUROCs; we have not retrieved it.

## What we propose

An **independent replication** of the model-specific contrast in a second screening
cohort, pre-registered before any data are seen.

| | |
|---|---|
| **primary endpoint** | the between-subgroup AUROC difference for a locked model: AUROC(active TB vs negatives without previous TB) − AUROC(with) |
| **minimum meaningful effect** | 0.05, fixed in advance |
| **secondary** | difference in differences against a commercial comparator on identical images; sensitivity at 70% specificity at a pre-locked threshold |
| **sample size** | **2,500–3,500 analysable**, covering a Bonferroni-adjusted requirement (2,182), a Kagujje-like 24% previous-TB fraction (2,139), and a 25% larger variance (2,291). Enrol ~3,500–4,500 for attrition |
| **locking** | weights, preprocessing and all thresholds frozen and hashed before the cohort is scored once; no change afterwards |

## Data we would need

De-identified frontal radiographs; microbiological result, ideally for **all** participants
regardless of AI output; previous-TB history **with an explicit unknown category** and the
ascertainment method recorded; a pseudonymous ID for de-duplication; scanner/site metadata
where permitted. Existing commercial CAD scores on the same images would be valuable but
are not essential.

## Governance

Ethics approval covering secondary AI analysis, a data-sharing agreement with authorship
terms agreed in advance, and de-identification of images and DICOM headers. **We can run
the analysis entirely inside the data holder's environment** — the pipeline needs under
1 GB of GPU memory and minutes of compute — so no patient-level data need move.

## Why this might be mutually useful

For a group that has already characterised this effect in commercial products, a
transparently reported research prototype adds a contrast they cannot get from vendors:
full access to the model, its training data composition, and its failure modes. For us it
is the only way to learn whether our observation is real or an artefact of one cohort.

**We are seeking a conversation and methodological feedback. A collaboration or any data
access would be a bonus, not an expectation.**
