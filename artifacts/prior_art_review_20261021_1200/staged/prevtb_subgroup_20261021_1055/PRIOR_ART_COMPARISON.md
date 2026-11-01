# Prior art: Kagujje et al. 2022, and a correction to our earlier characterisation

**Source re-verified in this run by direct retrieval of the full text.**

---

## 0. Correction, stated first

The preceding run (`realworld_generalization_20261021_1015`,
`07_PRIOR_ART_AND_CONTRIBUTION_ASSESSMENT.md`) claimed:

> "they report *thresholded specificity*, which a per-group threshold could in principle
> repair; **this run shows the deficit is present in AUROC**, so it is a ranking failure
> that recalibration cannot fix."

**That claim is wrong and is withdrawn here.** On re-reading the full text, Kagujje et al.
**do report AUROCs stratified by previous-TB status**, in Supplementary Table 2. The main
text states that *"Both CAD systems demonstrated an AUROC of at least 0.85 among all
participants"* and describes the stratified pattern qualitatively.

So measuring AUROC is **not** a methodological advance over their work. The earlier
characterisation overstated the distinction and should not be repeated. The prior run's
file was not edited — this run does not modify prior artifacts — so this document is the
correction of record.

## 1. What Kagujje et al. actually studied

> Kagujje M, Kerkhoff AD, Nteeni M, Dunn I, Mateyo K, Muyoyeta M. **"The Performance of
> Computer-Aided Detection Digital Chest X-ray Reading Technologies for Triage of Active
> Tuberculosis Among Persons With a History of Previous Tuberculosis."** *Clinical
> Infectious Diseases* 2022;76(3):e894–e901. DOI
> [10.1093/cid/ciac679](https://doi.org/10.1093/cid/ciac679) · PMID 36004409 · PMCID
> PMC9907528. Published, peer-reviewed.

| | |
|---|---|
| **setting** | public primary healthcare facility plus community screening points, Lusaka, Zambia, Jul 2017 – Dec 2018 |
| **design** | retrospective analysis of a cross-sectional active case-finding study |
| **flow** | 18,194 screened → 12,390 symptomatic adults ≥15 → 7,698 with digital CXR → **1,884 analysed** |
| **previous TB** | **452 (24.0%)**; **self-reported** ("ever treated for TB"); authors state *"verification of a self-reported history of previous TB using medical records was not undertaken"* |
| **reference standard** | sputum Xpert MTB/RIF; **culture not routinely available** |
| **prevalence** | overall 15.8% (298/1,884); **12.4% with previous TB**, **16.9% without** |
| **systems** | CAD4TB v7.0 (Delft), qXR v3.0 (Qure.ai) |
| **thresholds** | fixed at 15 (CAD4TB) and 6 (qXR), chosen for ~90% overall sensitivity |

**Headline thresholded results (95% CI):**

| system | previous TB | no previous TB |
|---|---|---|
| CAD4TB | sens 89.3% (78.1–96.0) · **spec 24.0% (19.9–28.5)** | sens 90.5% (86.1–93.3) · **spec 60.3% (57.4–63.0)** |
| qXR | sens 94.6% (85.1–98.9) · **spec 22.2% (18.2–26.6)** | sens 89.7% (85.1–93.2) · **spec 61.8% (58.9–64.5)** |

**Discrimination:** AUROC ≥ 0.85 overall for both systems. Stratified AUROCs are in
Supplementary Table 2, **which was not retrieved in this run**, so their values and
intervals are **UNVERIFIED here** and are not quoted. The main text's qualitative summary
is verified: **CAD4TB discriminated less well in people with previous TB, while qXR
performed comparably in both groups.**

**Authors' stated limitations:** facility-based sample with high HIV and symptom burden,
limiting generalisability; asymptomatic people excluded; smear status unavailable;
self-reported history may be misclassified because of stigma; Xpert less sensitive than
culture, and persistent Xpert positivity in previously treated patients may have produced
false positives; no human-reader comparison; subgroup-optimal thresholds were derived in
the same dataset and lack external validation.

## 2. Like-for-like comparison

| | Kagujje et al. 2022 | this project |
|---|---|---|
| cohort | Zambia, facility-based ACF, symptomatic adults | Vietnam, community ACF (Dryad) |
| n analysed | 1,884 | 2,708 |
| previous TB | 452 (24.0%), self-reported | 952 (35.2%), self-reported |
| prevalence | 15.8% | 10.3% |
| reference | Xpert MTB/RIF | Xpert Ultra |
| systems | two commercial CAD | one research pipeline + one commercial comparator (Lunit 3.9.0.1) |
| primary reported endpoint | **thresholded sensitivity/specificity**; AUROC also reported (supplementary) | **AUROC**, with a formal between-subgroup difference and interval |
| subgroup prespecified? | **yes** — it is the paper's subject | **no** — discovered in exploratory analysis |

**The clinical overlap is substantial**: both find that CAD performs worse in people
previously treated for TB, in an African and an Asian ACF cohort respectively, with
self-reported history and an Xpert reference standard.

**The methodological differences that actually matter**, neither of which makes our result
novel:

- **Endpoint.** Their headline is specificity at a fixed operating point; ours is a
  threshold-free AUROC gap with a confidence interval for the *difference*. These are
  different quantities and should not be compared numerically. A specificity gap of 36
  percentage points and an AUROC gap of 0.163 are not the same statistic and have no fixed
  conversion.
- **Comparative framing.** They compare two commercial products and find the effect is
  **product-dependent** — CAD4TB degraded, qXR did not. We compare one research pipeline
  against one commercial product on identical participants and find our degradation larger
  by 0.065 [0.040, 0.090]. **Their result is the more important of the two**: it shows the
  effect is not inevitable, which is also the most encouraging thing in this literature.
- **Prespecification.** Theirs was the study's purpose. Ours was found by screening five
  clinical variables and is hypothesis-generating.

## 3. What this project can and cannot claim

**Can claim:** an independent observation, in a different country and population and with
a different system, that is **consistent with** Kagujje et al., together with a
quantified comparison against a commercial product on identical participants.

**Cannot claim:**

- Novelty of the phenomenon. They published it first, prespecified, in a peer-reviewed
  journal.
- Novelty of the metric. They report stratified AUROCs too.
- That ours is the first ranking-level demonstration. Unverified — their supplementary
  table may already contain it. **Retrieve Supplementary Table 2 before writing anything
  that depends on this.**
- A comprehensive novelty assessment. **Two searches and one full-text retrieval is not a
  systematic review.** Other work on post-TB lung disease and CAD performance very likely
  exists and was not found here.

## 4. Unverified items

| item | status |
|---|---|
| Kagujje et al. Supplementary Table 2 stratified AUROCs | **NOT RETRIEVED** — do not characterise |
| prison-screening study said to show Lunit failing the WHO TPP in previous-TB patients | **NOT RETRIEVED** — found only as a search snippet; do not cite |
| whether any published work reports a between-subgroup AUROC *difference with an interval* | **UNKNOWN** — not searched systematically |
| whether Vietnam Xpert Ultra "trace" results were adjudicated | **UNKNOWN** — not stated in the Dryad documentation |

## 5. Consequence for the write-up

Frame the previous-TB result as **a replication in a second setting with a commercial
comparator**, cite Kagujje et al. as the prior finding in the same sentence, and **do not
present AUROC measurement as the contribution**. If a distinct contribution is wanted from
this strand, the defensible one is the **difference-in-differences against a certified
product on identical participants** — and even that needs a systematic check that nobody
has reported it.

## 6. Added 2026-10-21 — the cohort's own source publication

The Vietnam cohort used throughout this project is the **published test library of
Codlin et al., *PLOS Digital Health* 2025;4(4):e0000813** (DOI
10.1371/journal.pdig.0000813), released as Dryad 10.5061/dryad.5tb2rbpcv. That paper
already reports the previous-TB subgroup analysis for Lunit INSIGHT CXR v3.9.0.1 — the
comparator used here — on exactly these 2,708 participants.

Consequences for this document:

1. **The previous-TB question on this cohort is not open.** It is published, by the group
   that produced the data.
2. **Our analysis used a different construction.** Codlin et al. stratify every
   participant by `past_tb`; our earlier run shared all 280 positives across both strata.
   Under their construction our difference-in-differences is **+0.0481
   [−0.0170, +0.1113]** — it includes zero.
3. **Every description of this cohort in the project should cite Codlin et al. 2025.**
   None currently does.

Full detail: `prior_art_review_20261021_1200/PRIOR_ART_VERIFICATION.md`.
