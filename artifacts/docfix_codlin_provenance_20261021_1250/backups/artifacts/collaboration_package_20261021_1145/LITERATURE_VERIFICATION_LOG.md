# Literature verification log

All retrieval attempts made on **2026-10-21**. Status is recorded per source. Nothing is
characterised from a secondary summary or a search snippet.

---

## 1. Kagujje et al. 2022 — the key reference

> Kagujje M, Kerkhoff AD, Nteeni M, Dunn I, Mateyo K, Muyoyeta M. "The Performance of
> Computer-Aided Detection Digital Chest X-ray Reading Technologies for Triage of Active
> Tuberculosis Among Persons With a History of Previous Tuberculosis." *Clinical
> Infectious Diseases* 2022;76(3):e894–e901. DOI 10.1093/cid/ciac679. PMID 36004409.
> PMCID PMC9907528. **Peer-reviewed, published.**

| attempt | URL | outcome |
|---|---|---|
| 1 | `https://pmc.ncbi.nlm.nih.gov/articles/PMC9907528` | **SUCCESS** — full text read |
| 2 | `https://academic.oup.com/cid/article/76/3/e894/6675069` | **SUCCESS** — HTML full text accessible; PDF subscriber-only |
| 3 | `https://pmc.ncbi.nlm.nih.gov/articles/instance/9907528/bin/ciac679_supplementary_data.docx` | **FAILED** — returns a "Preparing to download…" interstitial; binary .docx not parseable by the fetch tool |
| 4 | `https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/cid/76/3/10.1093_cid_ciac679/1/ciac679_supplementary_data.docx` | **FAILED** — the live page's link carries signed query parameters that the base URL omits |

### Verified from the main text (quoted verbatim, confirmed in two independent sources)

> "Both CAD systems demonstrated an AUROC of at least 0.85 among all participants."

> **"The performance of the CAD4TB system was higher in individuals without previous TB
> compared with those with previous TB, while the performance of qXR was comparable in the
> 2 subpopulations."**

> "There was no difference in the AUROC values between HIV-positive and HIV-negative
> individuals."

Also verified: design (retrospective analysis of a cross-sectional ACF study, Lusaka,
Jul 2017–Dec 2018); flow 18,194 → 1,884 analysed; **452 (24.0%) with previous TB**,
**self-reported**, authors state *"verification of a self-reported history of previous TB
using medical records was not undertaken"*; reference standard sputum Xpert MTB/RIF,
culture not routinely available; prevalence 15.8% overall, 12.4% with previous TB, 16.9%
without; CAD4TB v7.0 and qXR v3.0 at fixed thresholds 15 and 6 for ~90% overall
sensitivity.

Thresholded results, verified: CAD4TB sens 89.3% (78.1–96.0) / **spec 24.0%
(19.9–28.5)** with previous TB, against sens 90.5% (86.1–93.3) / **spec 60.3%
(57.4–63.0)** without. qXR 94.6% (85.1–98.9) / **22.2% (18.2–26.6)** against 89.7%
(85.1–93.2) / **61.8% (58.9–64.5)**.

### UNVERIFIED

**The exact stratified AUROC values in Supplementary Table 2 remain inaccessible** after
the four attempts above. Their values, confidence intervals and denominators are **not
known** and are not estimated, inferred or copied from any secondary source anywhere in
this package.

**Note on an unreliable secondary summary.** A web-search summary generated during this
run asserted that Kagujje et al. "reports threshold-based sensitivity and specificity
rather than AUROC". **That is incorrect** — the article text cites Supplementary Table 2
for stratified AUROCs. This is recorded as a concrete reason to rely only on direct
retrieval.

## 2. Qin et al. 2025 — multi-product and radiologist comparison

> Qin ZZ, Van der Walt M, Moyo S, Ismail F, Maribe P, Denkinger CM, Zaidi S, Barrett R,
> Mvusi L, Tsibolane Y, Mkhondo N, Zuma K, Manda S, Koeppel L, Mthiyane T, Creswell J.
> "Comparing the accuracy of computer-aided detection (CAD) software and radiologists from
> multiple countries for tuberculosis detection in chest X-Rays." *Scientific Reports*
> 2025;15:22540. DOI 10.1038/s41598-025-06164-w. **Peer-reviewed, published.**
> Retrieved: `https://pmc.ncbi.nlm.nih.gov/articles/PMC12215708/` — **SUCCESS**.

Verified: 774 chest X-rays from the South African National TB Prevalence Survey; 12 CAD
products and 11 radiologists from India, Nigeria, the UK and the US; **197 participants
(26%) with prior TB**; overall CAD AUCs reported (e.g. Lunit 0.902).

**It does NOT report prior-TB-stratified AUROC per CAD product.** It defers CAD subgroup
results to a complementary publication and states that most CAD products showed
*"significantly worse performance in PLHIV, individuals with prior TB, and older adults."*

Verified and directly relevant: **radiologists also degrade.** *"All groups demonstrated
higher sensitivity but significantly lower specificity in individuals with a prior TB"* —
e.g. UK radiologists 84.6% sensitivity (75.8–90.6) in prior-TB participants. Prior-TB
subgroup denominators for the radiologist analysis are not given.

## 3. The critical unretrieved lead

> Qin et al., *Lancet Digital Health*, 2024 — cited as reference 17 of the Scientific
> Reports paper above and described there as the complementary external validation of CAD
> on the South African National TB Prevalence Survey.

**NOT RETRIEVED. Status: unknown.** This is the single most important gap in this
package, because it is the publication most likely to already contain **per-product
AUROC stratified by prior TB** — which is the form of our own observation. **Retrieve it
before writing anything that frames the model-comparator contrast as new.**

## 4. Other leads, recorded with status

| source | status | relevance |
|---|---|---|
| Qin ZZ et al., five-AI-algorithm evaluation, Bangladesh — `https://arxiv.org/pdf/2006.05509` | **NOT RETRIEVED; preprint (arXiv, 2020)** | A search snippet indicates it reported that algorithm performance "differed between age groups, patient sources, and prior TB history" and noted poor discrimination of "old scarring" from active lesions. If accurate, **algorithm-specific differences by prior-TB history were described as early as 2020.** Must be retrieved before any model-specificity claim. Preprint status to be checked for a journal version. |
| qure.ai "evidence" index pages | **vendor summary, not a primary source** | Used only to locate primary citations. Nothing is cited from them. |
| Prison-screening diagnostic accuracy study (2023) | **NOT RETRIEVED** | Earlier runs referenced it from a snippet. Still uncited and uncharacterised. |
| Eur Respir J 2024 (Edem), `archive.lstmed.ac.uk/25000/7/` | **NOT RETRIEVED** | Appeared in search; relevance unconfirmed. |
| Paediatric CAD4TB study (2024) | **NOT RETRIEVED** | Snippet indicates it does not address prior TB. Not pursued. |

## 5. Scope statement

**This is a targeted check, not a systematic review.** Three web searches and four
full-text retrieval attempts were made. No database query was run against PubMed, Embase
or Scopus; no citation chaining was performed beyond following reference 17; no screening
protocol was applied. **No "first", "novel" or "only" claim anywhere in this package
rests on this log**, and §3 identifies a specific publication that could overturn the one
observation that currently looks distinctive.
