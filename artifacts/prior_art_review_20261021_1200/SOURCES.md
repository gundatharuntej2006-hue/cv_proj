# Sources and access status

All attempts made 2026-10-21. **Nothing below is characterised from a search snippet or a
secondary summary.** Where retrieval failed, the values are marked unverified and are not
estimated.

---

## Retrieved and read

| # | source | URL | status |
|---|---|---|---|
| 1 | **Codlin AJ, Vo LNQ, Dao TP, Forse RJ, Dang HTM, Nguyen LH, Nguyen HB, Dinh LV, Sidney Annerstedt K, Lundin J, Lönnroth K.** "Comparison of different Lunit INSIGHT CXR software versions when reading chest radiographs for tuberculosis." *PLOS Digital Health* 2025;4(4):e0000813. DOI 10.1371/journal.pdig.0000813. Published 15 Apr 2025. | https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000813 | **RETRIEVED, full text, open access** |
| 2 | **Kagujje M, Kerkhoff AD, Nteeni M, Dunn I, Mateyo K, Muyoyeta M.** *Clin Infect Dis* 2022;76(3):e894–e901. DOI 10.1093/cid/ciac679. PMID 36004409. | https://pmc.ncbi.nlm.nih.gov/articles/PMC9907528 and https://academic.oup.com/cid/article/76/3/e894/6675069 | **RETRIEVED, main text, two independent sources** |
| 3 | **Qin ZZ, et al.** "Computer-aided detection of tuberculosis from chest radiographs in a tuberculosis prevalence survey in South Africa." *Lancet Digital Health* 2024;6(9):e605–e613. DOI 10.1016/S2589-7500(24)00118-3. | https://pmc.ncbi.nlm.nih.gov/articles/PMC11339183/ | **RETRIEVED, main text via PMC.** Publisher URL `thelancet.com/.../PIIS2589-7500(24)00118-3/fulltext` returned **HTTP 403** |
| 4 | **Qin ZZ, et al.** "Comparing the accuracy of computer-aided detection (CAD) software and radiologists from multiple countries for tuberculosis detection in chest X-Rays." *Sci Rep* 2025;15:22540. DOI 10.1038/s41598-025-06164-w. | https://pmc.ncbi.nlm.nih.gov/articles/PMC12215708/ | **RETRIEVED, full text** |
| 5 | **Dryad dataset** for source 1: "Comparison of different Lunit INSIGHT CXR software versions…" DOI 10.5061/dryad.5tb2rbpcv | local copy at `data/raw/vietnam/` with its README | **HELD LOCALLY AND READ** |

## Retrieval attempted and failed

| source | URL attempted | outcome |
|---|---|---|
| Kagujje et al. Supplementary Table 2 (stratified AUROCs) | `https://pmc.ncbi.nlm.nih.gov/articles/instance/9907528/bin/ciac679_supplementary_data.docx` | **FAILED** — "Preparing to download…" interstitial; binary .docx not parseable |
| same | `https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/cid/76/3/10.1093_cid_ciac679/1/ciac679_supplementary_data.docx` | **FAILED** — live link carries signed query parameters absent from the base URL |
| Qin et al. 2024 appendix pp 20–24 (per-product prior-TB AUCs) | `https://pmc.ncbi.nlm.nih.gov/articles/instance/11339183/bin/mmc1.pdf` | **FAILED** — same PMC interstitial; 3.5 MB PDF not parseable |
| Qin et al. 2024 publisher full text | `https://www.thelancet.com/journals/landig/article/PIIS2589-7500(24)00118-3/fulltext` | **FAILED** — HTTP 403 Forbidden |

**Four distinct supplementary/full-text retrievals failed.** Their contents are treated as
unknown throughout this review.

## Identified but not retrieved

| source | why it matters | status |
|---|---|---|
| Qin ZZ et al., five-algorithm evaluation, Bangladesh, arXiv 2006.05509 (2020, **preprint**) | a snippet indicates it reported performance differing by prior-TB history across algorithms, which would push algorithm-specific findings back to 2020 | **NOT RETRIEVED.** Preprint; journal version not checked |
| Prison-screening diagnostic accuracy study (2023) | referenced in earlier runs from a snippet only | **NOT RETRIEVED**, still uncited |

## Unreliable secondary summaries encountered

A web-search summary during this review asserted that Kagujje et al. "reports
threshold-based sensitivity and specificity rather than AUROC". **That is false** — the
article cites Supplementary Table 2 for stratified AUROCs. A second summary described
Codlin et al.'s figures without noting that the no-past-TB AUCs were identical between
versions. Both are recorded as concrete reasons to rely only on direct retrieval.

## Scope statement

**This is a targeted prior-art check, not a systematic review.** Five web searches and
nine retrieval attempts across four papers. No structured database query (PubMed, Embase,
Scopus), no screening protocol, no citation chaining beyond following one reference. The
three papers examined are sufficient to **reject** a novelty claim; they are not
sufficient to establish that anything is new.
