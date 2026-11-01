# Data rights assessment — Vietnam Dryad cohort

**Assessed 2026-10-23 from primary sources. This supersedes the "unresolved" blocker
recorded in `artifacts/kaggle_concept_raydino_20261021_1830/`.**

---

## Status: **CLEAR FOR THE PROPOSED PUBLIC-DATA RESEARCH USE**

Confidence: **high** for the copyright/database-rights question. Two independent official
sources agree, one of them dataset-specific and machine-readable.

---

## 1. What was checked, and what each source said

### 1a. Dryad's archive-wide reuse policy — **retrieved 2026-10-23**

<https://v3.datadryad.org/help/guides/reuse>

> "All datasets on Dryad are published under the CC0 (Creative Commons Zero) Public
> Domain Dedication."

The page also states you "can reuse, modify, share, and even redistribute the data" for
commercial and non-commercial purposes, that attribution is **not legally required** under
CC0 but is strongly recommended as scholarly courtesy, and it records **no exceptions** and
no dataset class carrying different terms.

### 1b. The dataset landing page — **retrieved 2026-10-23**

<https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbpcv>

Displays **no CC0 badge and no dataset-specific License field**. The only licence-adjacent
text is the site footer, "Content on this site is licensed for reuse", linking to
`/terms#copyright`. **This absence is what produced the earlier blocker**, and on its own
it is weak evidence either way.

Thirteen files, 20.85 GB total: eleven DICOM zips plus the participant CSV and README.
No ethics statement, no consent statement, no usage restriction. The README says
"de-identified participant data", and the variable notes record that fields were withheld
"to avoid inclusion of more than three indirect identifiers".

### 1c. The dataset-specific API record — **decisive, retrieved 2026-10-23**

<https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.5tb2rbpcv>

```
"license": "https://spdx.org/licenses/CC0-1.0.html"
"title":   "Comparison of different Lunit INSIGHT CXR software versions..."
"versionNumber": 5
"versionStatus": "submitted"
"publicationDate": "2025-03-14"
```

**This is a dataset-specific, machine-readable, version-specific CC0-1.0 declaration for
this exact record.** No `rights`, `rightsURI` or contradictory field is present.

### 1d. The README and the article

The README states no licence — which is what triggered the original caution. **A README
omitting a licence line is not a licence restriction.** The deposit's licence lives in the
repository metadata, and it says CC0-1.0.

## 2. The reasoning

| question | answer |
|---|---|
| Does an archive-wide policy exist? | Yes — "All datasets on Dryad are published under CC0" |
| Does it apply to **this** record and version? | Yes — the API returns CC0-1.0 for this DOI at version 5 |
| Any dataset-specific notice overriding it? | **None found** on the landing page, in the README, or in the metadata |
| Any third-party or ethics constraint stated? | **None stated.** The data are described as de-identified with indirect identifiers limited |
| Does CC0 permit redistribution? | **Yes, explicitly** — Dryad's own guidance says "even redistribute" |

The earlier blocker rested on absence of evidence (no licence line in a README). The API
record is **evidence of presence**, is dataset-specific, and agrees with the archive-wide
policy. The blocker is withdrawn.

## 3. What this does **not** settle

1. **CC0 waives copyright and database rights. It does not and cannot waive
   human-subjects obligations.** These are de-identified chest radiographs of real
   patients. CC0 is a statement about intellectual property, not an ethics approval and
   not a finding that re-identification risk is zero. Treat the images as sensitive
   regardless of their licence.
2. **It says nothing about model weights.** RayDINO remains under the **FAIR
   Noncommercial Research License** and gated; CLEAR is Apache-2.0. Licensing of data and
   of models are separate questions with separate answers — see
   `MODEL_LICENCE_ASSESSMENT.md`.
3. **It does not oblige us to upload anything.** CC0 permits redistribution; prudence is a
   separate judgement. **Recommendation: if the images are placed on Kaggle, make the
   dataset Private.** Nothing is lost by doing so and it keeps the re-identification
   surface minimal.
4. **One metadata observation, recorded not over-read:** the API reports
   `"versionStatus": "submitted"` for version 5 while also giving a 2025-03-14 publication
   date, with files publicly listed and downloadable. This is noted for completeness; it
   does not appear to qualify the licence, and no conclusion is drawn from it.

## 4. Consequence for this project

The concept audit is **no longer blocked on data rights**. The local copy at
`data/processed/v1/vietnam_img512` may be used for research analysis, and may in principle
be uploaded to a private Kaggle dataset.

No author contact is required, so **no email draft was prepared and none was sent.** The
Dryad record does list contactable authors should a question arise later.

## 5. Access log

| source | URL | retrieved |
|---|---|---|
| Dryad reuse guidance | `v3.datadryad.org/help/guides/reuse` | 2026-10-23 |
| Dataset landing page | `datadryad.org/dataset/doi:10.5061/dryad.5tb2rbpcv` | 2026-10-23 |
| Dataset API record | `datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.5tb2rbpcv` | 2026-10-23 |
| Associated article | Codlin et al., *PLOS Digit Health* 2025;4(4):e0000813, doi:10.1371/journal.pdig.0000813 | 2026-10-21 |
