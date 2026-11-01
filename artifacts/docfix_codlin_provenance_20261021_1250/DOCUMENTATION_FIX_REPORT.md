# Documentation fix: cohort provenance and superseded outreach

**Run:** `artifacts/docfix_codlin_provenance_20261021_1250/` · 2026-10-21, 13:49 → 14:10
**HEAD `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, unchanged.** Documentation only.

**No training, scoring, feature extraction, threshold fitting or `internal_test` access.
No model code, checkpoint, embedding, prediction file or manifest was modified — 173
protected artifacts verified unchanged by hash. No email sent, no contact detail sought
or invented. No commit, no push.**

---

## 1. What was verified first

### The citation, from two independent sources

> Codlin AJ, Vo LNQ, Dao TP, Forse RJ, Dang HTM, Nguyen LH, Nguyen HB, Dinh LV,
> Sidney Annerstedt K, Lundin J, Lönnroth K. **"Comparison of different Lunit INSIGHT CXR
> software versions when reading chest radiographs for tuberculosis."** *PLOS Digital
> Health* 2025;**4**(4):e0000813.
> [doi:10.1371/journal.pdig.0000813](https://doi.org/10.1371/journal.pdig.0000813).
> Received 8 Jul 2024, accepted 5 Mar 2025, published 15 Apr 2025.

The **Dryad record** ([10.5061/dryad.5tb2rbpcv](https://doi.org/10.5061/dryad.5tb2rbpcv),
published 14 Mar 2025) independently confirms the identical title and the **same eleven
authors in the same order**. This resolved an ambiguity from an earlier retrieval, where
one rendering of the article page omitted Lönnroth — he is an author.

**One thing the Dryad record does not do:** its "related works" section is empty, so it
carries no formal link to the publication. The association rests on the identical title,
the identical author list, and the statistical reproduction below.

### Cohort identity, by statistical reproduction

Preserved from `artifacts/prior_art_review_20261021_1200/`: our local copy reproduces
**all six of the paper's published Lunit AUROCs to two decimal places** — overall 0.76 /
0.78, previous TB 0.67 / 0.73, no previous TB 0.81 / 0.81 for v3.1.0.0 and v3.9.0.1 — and
matches n = 2,708, 10.3% Xpert Ultra positivity and 35.2% reporting previous TB.

**This establishes identity by statistical reproduction, not by patient-ID matching**,
which the de-identified release makes impossible. Every document that now cites the paper
says so.

### The corrected comparative result

Preserved unchanged: under the paper's **full stratification** construction, the
prototype-versus-Lunit difference in differences is **+0.0481, 95% CI [−0.0170, +0.1113]**.
**The interval includes zero; this does not establish differential degradation.** The
subgroup was not pre-specified.

### Distinctions maintained throughout

AUROC/ranking, thresholded sensitivity and specificity, and calibration are kept separate
in every edit. Nowhere is it suggested that recalibration or a subgroup threshold can
repair an AUROC deficit — it cannot, because AUROC is threshold-free.

**No claim of a comprehensive novelty search is made anywhere.** The review was targeted;
two supplementary sources remain unretrieved.

## 2. Files changed — 10

Full detail in `tables/claim_change_log.csv` (22 rows). Before-and-after hashes in
`tables/edited_file_hashes.csv`. Pre-edit copies in `backups/`.

### Tracked documentation (5)

| file | change |
|---|---|
| `docs/p2_vietnam_result.md` | provenance block with the full citation, Dryad DOI, the reproduction basis, the no-related-works note, and the fact that **the Lunit comparator used throughout is that paper's published comparator**; a "what remains unknown" block; three "consecutive" descriptions corrected; the unverified **CC0 licence assertion removed** |
| `report/paper_draft.md` | footnote citation on the dataset table; "consecutive screening" → "programmatic active case finding" in two places; abstract wording; "all participants" → "all participants in the test library" |
| `report/p2_00_abstract_intro.md` | footnote citation; two "consecutive" corrections |
| `report/p2_99_conclusion.md` | one "consecutive" correction |
| `docs/p2_external_validation.md` | citation added where the screening figure is quoted |

### Run artifacts (5)

`OUTREACH_EMAIL_DRAFT.md` and `ONE_PAGE_CONCEPT_NOTE.md` now carry **DO NOT SEND —
SUPERSEDED** banners; `CLAIM_BOUNDARIES.md` points to its revised successor;
`LITERATURE_VERIFICATION_LOG.md` and `PRIOR_ART_COMPARISON.md` record Codlin et al. and
the reproduction. Applied from the patch prepared in the prior run
(`patch -p1`, exit 0).

## 3. Edits deliberately not made

The brief asked that unnecessary edits be recorded rather than manufactured.

| suggested edit | finding |
|---|---|
| correct claims that the previous-TB effect is novel | **No such claim exists in tracked documentation.** An automated scan of all of `report/` and `docs/` returns zero hits. That analysis lives only in run directories, where it was already correctly qualified |
| correct "patch tokens are generally discarded" | **Zero hits.** The claim was made in conversation only and was already recorded as an error in `artifacts/p2/audit/SOURCES.md` |
| correct "our model degrades significantly more than Lunit" | **Zero hits** in tracked documentation |

## 4. Consistency check results

`scripts/consistency_check.py`, output `tables/consistency_check.json`.

| check | result |
|---|---|
| banned-claim scan over all of `report/` and `docs/` (6 patterns) | **0 hits** |
| Codlin citation + DOI + Dryad DOI present in all four cohort-describing files | **4 / 4 pass** |
| "statistical reproduction, not ID matching" qualifier present | **yes** |
| "DiD includes zero / inconclusive" qualifier present | **yes** |
| protected artifacts unchanged | **173 checked, 0 changed** |

The checker itself needed two corrections during the run: it compared against hard-wrapped
text without normalising whitespace, and did not strip blockquote markers, so it reported
a journal name as missing when it was present but line-wrapped. Both fixed; neither
affected a document.

## 5. Outstanding, and why

Two files contain descriptions of the cohort as "consecutive" and lack the citation, but
**are outside the list of files this task authorised for editing**, so they were left
alone and are reported instead:

| file | text |
|---|---|
| `report/slides/p2_slides_outline.md` | *"\| **Vietnam** \| **consecutive screening cohort** \|"* and *"0.6151 on 2,708 consecutive screening attendees"* |
| `docs/p2_focal_pooling_final.md` | *"raises performance on a consecutive TB screening cohort"* and *"Our evaluation adds a consecutive screening cohort with an Xpert reference standard"* |

**Recommended follow-up:** apply the same two corrections there — "consecutive" →
"screening (programmatic ACF)", plus the citation. Until then the repository is
internally inconsistent on this point.

Remaining uses of "consecutive" elsewhere are **generic methodological advice** ("one
consecutive cohort is worth more than three case-control sets", "at least one consecutive
cohort, or an explicit statement that none was tested") or occur inside the new corrective
text itself. Those are correct as written.

## 6. What remains unknown

- **The literature check was targeted, not systematic.** Sufficient to establish that the
  previous-TB effect is already described — Kagujje et al. 2022, Qin et al. 2024, Codlin
  et al. 2025 — and therefore **not novel here**. Not sufficient to establish that
  anything is new.
- **Product variation is real and must not be flattened.** Kagujje et al. found CAD4TB
  discriminated worse with previous TB while **qXR was comparable across subgroups**.
- **Two supplementary sources remain unverified:** Kagujje et al. Supplementary Table 2
  and Qin et al. 2024 appendix pp 20–24. Values are neither quoted nor estimated.
- **RAD-DINO checkpoint-selection exposure is unknown** — the model card does not name
  those datasets.
- **No causal explanation** of the subgroup degradation is established.
- **The Dryad licence is not stated** on the record; the previous "CC0" assertion has been
  removed rather than replaced with another guess.
- **Enrolment completeness is undocumented**: whether everyone mobilised was imaged and
  tested, and whether Xpert Ultra trace results were adjudicated, are not stated.

## 7. Definition of done

| requirement | status |
|---|---|
| repository cites Codlin et al. for the source cohort | **done** — 4 / 4 cohort-describing files, 2 outstanding outside scope |
| previous-TB observation not presented as novel | **done** — no such claim existed in tracked docs; explicit non-novelty statement added to `docs/p2_vietnam_result.md` |
| model-versus-Lunit difference not overclaimed | **done** — recorded as +0.0481 [−0.0170, +0.1113], inconclusive |
| outreach package clearly superseded | **done** — DO NOT SEND banners applied |
| no model scores or production code changed | **done** — 173 protected artifacts hash-verified unchanged |
| insufficient evidence left qualified, not guessed | **done** — licence, enrolment completeness, trace adjudication and two supplements all marked unresolved |
