# Phase 4 — Prior art and contribution assessment

**Headline: the central finding of this run is a replication of established clinical
literature, not a discovery.** That is stated first because it is the most important thing
in this document.

---

## 1. The paper that already reports this finding

**VERIFIED — full text retrieved and read in this run.**

> Kagujje M, Kerkhoff AD, Nteeni M, Dunn I, Mateyo K, Muyoyeta M. **"The Performance of
> Computer-Aided Detection Digital Chest X-ray Reading Technologies for Triage of Active
> Tuberculosis Among Persons With a History of Previous Tuberculosis."** *Clinical
> Infectious Diseases* 2022;76(3):e894–e901. DOI
> [10.1093/cid/ciac679](https://doi.org/10.1093/cid/ciac679). PMID 36004409. PMCID
> PMC9907528. Published.

| | |
|---|---|
| **design** | retrospective analysis of a cross-sectional active case-finding study, Lusaka, Zambia, Jul 2017 – Dec 2018 |
| **cohort** | 18,194 screened → 1,884 analysed; **452 with previous TB**, 1,432 without; 298 (15.8%) microbiologically confirmed |
| **systems** | CAD4TB v7.0 (Delft), qXR v3.0 (Qure.ai), thresholds fixed for ~90% overall sensitivity |
| **reference standard** | sputum Xpert MTB/RIF; culture not routinely available |
| **key result** | **CAD4TB specificity 24.0% (19.9–28.5) with previous TB vs 60.3% (57.4–63.0) without.** qXR 22.2% (18.2–26.6) vs 61.8% (58.9–64.5). Sensitivity preserved in both groups (89–95%) |
| **caveats the authors state** | analytic sample restricted to symptomatic adults with complete data, so not fully consecutive; previous TB self-reported, not record-verified |

**Overlap with this project: near-total on the phenomenon.** They found it first, in a
larger and more clinically rigorous study, across two commercial products.

**What differs here, and it is a refinement rather than a discovery:**

1. **They report specificity at a fixed threshold; this run reports AUROC.** That
   distinction matters operationally. A specificity gap at one threshold can in principle
   be a calibration artefact, fixable by a per-group threshold. **An AUROC gap cannot** —
   it is a ranking failure. Measured here: 0.7811 against negatives without past TB,
   **0.6182** against negatives with it (n = 1,130; 280 TB). So the problem survives
   recalibration, which the fixed-threshold framing leaves open.
2. **A commercial comparator on identical patients** degrades on the same axis but less:
   Lunit 3.9.0.1 0.8170 → **0.7192**, a drop of 0.098 against our 0.163. The Kagujje study
   compares two commercial products to each other; it has no research system alongside.
3. **A frozen foundation-model pipeline** rather than purpose-built commercial CAD.

**It remains a replication.** Anyone citing this project's past-TB result must cite
Kagujje et al. as the prior finding.

## 2. Corroborating evidence found, not independently verified

**PARTIALLY VERIFIED — search result snippets only; full texts not retrieved.** Methods and
results below are therefore *not* characterised beyond what the snippet stated.

- A prison-screening diagnostic accuracy study (2021 data, published 2023) reportedly found
  that at 90% sensitivity only Lunit and qXR met the WHO TPP overall, and that **even Lunit
  failed the TPP for individuals with previous TB**. Source listed via qure.ai's evidence
  index; the underlying publication was not retrieved. **Do not cite from this run.**
- An ultra-portable X-ray carceral ACF study
  ([PMC12581292](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12581292/)) reportedly imaged
  all participants irrespective of symptoms with pooled Xpert Ultra. Not retrieved.

## 3. Prior art on the architectural side, carried forward unchanged

From `artifacts/p2/audit/SOURCES.md`, re-stated with its original verification status. **No
new retrieval was performed for these in this run.**

| source | status | bearing |
|---|---|---|
| **RADAR** — *Acquisition-Adversarial Attention Pooling over a Frozen Chest-Radiograph Foundation Model for Tuberculosis Screening*, MICCAI 2026 satellite | **retrieved and read in an earlier run** | Closest architectural prior art. Frozen CXR foundation model, learned attention over patches, adversarial acquisition objectives. **The regional-pooling architecture here is not novel.** |
| **RAD-DINO model card** | **retrieved and read** | Backbone provenance; checkpoint-selection datasets unnamed, so exposure to public TB benchmarks is **unknown** |
| **CheXFound / GLoRI**, IEEE TMI 44(12) 2025 | **NOT RETRIEVED** | cited only as context that global+local integration is an established strategy; no specific claim rests on it |
| *Domain-aware LoRA-adapted CXR foundation model for TB*, MICCAI 2026 satellite | page retrieved, **bibliographic details unconfirmed** | excluded from the paper draft |
| *Shortcut learning in medical AI*, npj Digit Med 2024 | **NOT RETRIEVED** | not relied on |

## 4. Is there a defensible contribution left?

**Not a novel architecture.** Frozen radiology embeddings, regional pooling, linear probes,
calibration and domain-generalisation methods are all established, and RADAR covers the
closest TB-specific combination while adding mitigation this project lacks.

**Not a novel clinical finding.** The past-TB specificity collapse is Kagujje et al. 2022.

**What survives as a genuine, if modest, contribution:**

1. **A matched, measured audit across the full benchmark-to-screening ladder using one
   frozen pipeline.** TBX11K dev 0.9368 → Shenzhen/Montgomery 0.9690 → DA 0.9755 / DB
   0.9497 → Vietnam 0.7242, with a commercial comparator on identical patients at the
   screening rung and every configuration change measured on the same cohorts. The value
   is the *controlled* comparison, not any single number.
2. **The ranking-versus-calibration decomposition of the past-TB failure.** Showing the
   deficit persists in AUROC, not only in fixed-threshold specificity, says something the
   prior literature leaves open and rules out the obvious remedy.
3. **A catalogue of interventions that did not work, each with an interval.** Scanner
   subspace removal (rejected, −0.0002), scanner-balanced training (impossible — no source
   labels), per-scanner offsets (inconclusive, +0.0206 [−0.0030, +0.0436]), extra
   non-TB-abnormal training negatives (this run: −0.0136, and a prior replication),
   attention-MIL (worse), more training data (flat). Negative results with
   preregistrations and effect sizes are publishable and are what this project actually
   has.
4. **A small, mechanistically clear label-definition effect**: treating `latent_tb` as
   negative gives +0.029 [+0.0223, +0.0363] on the past-TB subgroup. Below the +0.05 this
   run declared meaningful, so **a lead, not a result.**

**Honest framing for the write-up:** this is a **replication and benchmarking study with a
preregistered negative-results catalogue**, not a methods paper. Its strongest claim is
about *evaluation practice* — that curated case-control AUROC is an actively misleading
guide to screening performance, demonstrated across five cohorts with matched
configurations — rather than about any model.

## 5. Gaps genuinely still open in the literature

Stated as gaps, not as claims that this project has filled them:

- Whether the past-TB deficit is a **ranking** failure for commercial CAD too, as it is
  here. Kagujje et al. report thresholded specificity only.
- Whether **training-label definition** (active vs any TB signs) accounts for a material
  share of it. This run gives a first, small, exploratory estimate on one exposed cohort.
- Whether any mitigation **transfers to an unseen screening site**. Nothing in this project
  or, as far as this run's search reached, in the retrieved literature demonstrates that.
