# Run manifest — prior-art review, 2026-10-21

**Directory:** `artifacts/prior_art_review_20261021_1200/`
**Start 12:29 · end 12:47 local · ≈18 minutes**
**HEAD `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, unchanged start to end**

Document and literature review only. **No training, no model scoring, no threshold
fitting, no access to TBX11K `internal_test`, no modification of model code, weights,
prediction files or any existing result. No commits, no pushes. No message sent to anyone
and no contact detail looked up, inferred or invented.**

---

## Environment

Python 3.13.2 · numpy 2.2.6 · scikit-learn 1.9.0 · Windows-11-10.0.26200-SP0.
CPU only; no GPU was used.

## Inputs read, never written

| file | sha256 |
|---|---|
| `data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv` | `ff98c9df9dd9d70f3aa1ea8bd4a4b4d1b45ba3b3c1e05ba3cf4f63d3b7bbbd7f` *(recomputed at run time; see `tables/cohort_identity_and_reanalysis.json` for the authoritative value)* |
| `artifacts/overnight_scanner_run_20261021_0208/tables/vietnam_case_scores.csv` | recorded in `tables/cohort_identity_and_reanalysis.json` |
| `artifacts/collaboration_package_20261021_1145/*` | unchanged; hashes in that directory's `PACKAGE_MANIFEST.sha256` |
| `artifacts/prevtb_subgroup_20261021_1055/*` | unchanged; hashes in that directory's `FILE_MANIFEST.md` |

The authoritative input hashes for this run are in
`tables/cohort_identity_and_reanalysis.json` → `inputs`.

## Commands

```
py -3 artifacts/prior_art_review_20261021_1200/scripts/cohort_identity_and_reanalysis.py
py -3 artifacts/prior_art_review_20261021_1200/scripts/make_patch.py
patch --dry-run -p1 -i artifacts/prior_art_review_20261021_1200/PROPOSED_OUTREACH_REVISIONS.patch
py -3 artifacts/prior_art_review_20261021_1200/scripts/make_run_manifest.py
```

Literature retrieval used the web-fetch tool against the URLs listed in `SOURCES.md`.
Nine retrieval attempts across four papers; four failed, all documented.

## Key results produced

| result | value |
|---|---|
| cohort identity vs Codlin et al. 2025 | **all six published AUCs reproduce at 2 dp**; n, positivity and previous-TB fraction all match |
| our prototype, past TB (construction A) | 0.6392 [0.5856, 0.6922] |
| our prototype, no past TB (construction A) | 0.7720 [0.7403, 0.8025] |
| our degradation (A) | **+0.1328 [0.0718, 0.1951]** — excludes zero |
| Lunit v3.9.0.1 degradation (A) | +0.0847 [0.0221, 0.1495] — excludes zero |
| **difference in differences (A)** | **+0.0481 [−0.0170, +0.1113] — INCLUDES ZERO** |
| difference in differences (B, earlier construction) | +0.0651 [0.0399, 0.0900] |

Bootstrap: 10,000 replicates, seed 20261021. Under construction A the two strata are
independent, so each is resampled separately; the difference-in-differences resamples all
four stratum-by-class cells per replicate.

## Deliverables

| file | contents |
|---|---|
| `PRIOR_ART_VERIFICATION.md` | the three papers, cohort-identity proof, reanalysis, recommendation |
| `PRIOR_ART_MATRIX.csv` | 11 rows separating established phenomenon / this cohort's own paper / our additions / limitations |
| `CLAIM_BOUNDARIES_REVISED.md` | supersedes the previous boundaries; two claims demoted |
| `PROPOSED_OUTREACH_REVISIONS.patch` | 5 files, additive annotations, **not applied**; `patch --dry-run -p1` exit 0 |
| `SOURCES.md` | every URL with access status, including four documented failures |
| `RUN_MANIFEST.md` | this file |
| `tables/cohort_identity_and_reanalysis.json` | machine-readable results and input hashes |
| `staged/` | the five edited files as they would look if the patch were applied |
| `scripts/` | two scripts, both read-only with respect to existing artifacts |

## Verification that nothing was modified

Both earlier run directories carry their own hash manifests, and both verify unchanged:

```
artifacts/prevtb_subgroup_20261021_1055/FILE_MANIFEST.md          14 files, 0 discrepancies
artifacts/collaboration_package_20261021_1145/PACKAGE_MANIFEST.sha256   9 files, 0 discrepancies
```

`git status --short` shows the same six tracked files modified by an earlier
documentation task — preserved untouched — plus untracked run directories only.

## Scope caveat

**This is a targeted prior-art check, not a systematic review.** Five web searches, nine
retrieval attempts, three papers read in full plus one supporting paper. Sufficient to
**reject** a novelty claim; not sufficient to establish that anything is new.

## Output hashes

SHA-256 of every file this run created, excluding `RUN_MANIFEST.md` itself.

| file | bytes | sha256 |
|---|---|---|
| `CLAIM_BOUNDARIES_REVISED.md` | 6135 | `6a15037766d378d9ef5d76fc85d909c5b813cb13ee64d9b5a072b42d7cdc1fd7` |
| `PRIOR_ART_MATRIX.csv` | 5089 | `5e9626754ff98eefdbfd7e8630d8320bf53ba977660ca02fa8fa109cbd058d9f` |
| `PRIOR_ART_VERIFICATION.md` | 10478 | `d296c59719bd38f30dfbba6655c261e6862d688553bab0020bb5bbce25f27ba7` |
| `PROPOSED_OUTREACH_REVISIONS.patch` | 7301 | `e8bb4c68d7653a17b26435e8705b6c3692c7df8eb88c9dfa450b97955e5db475` |
| `scripts/cohort_identity_and_reanalysis.py` | 11015 | `6b54ecbd734b513bbac709b6d81a1b8da8daff01c2bcbe7a540a88f224564fbd` |
| `scripts/make_patch.py` | 8269 | `432ec838c4492fd3b5b8f9ef5bfada06c5b7d5250ae2f02042c4858e486a25cd` |
| `scripts/make_run_manifest.py` | 1109 | `862bf4126f7622915834539161b9604e6f36dee4fc10c03e336c303e444fefcc` |
| `SOURCES.md` | 4486 | `7afd39d986203a981db1f065d8a4cbf1755642b2cbbfbc74ecf66bd99d4b6bd7` |
| `staged/collaboration_package_20261021_1145/CLAIM_BOUNDARIES.md` | 7599 | `2fcd4ca9b98d02aa34f5a05a955ca034abda6a46dfa76e1d64f5710e4bc69be3` |
| `staged/collaboration_package_20261021_1145/LITERATURE_VERIFICATION_LOG.md` | 8316 | `6c9917a3188b40f8e32bd1aa50789abe3ec5e5f4cc9c8d0bee9e1a59fecaf260` |
| `staged/collaboration_package_20261021_1145/ONE_PAGE_CONCEPT_NOTE.md` | 6038 | `62d4c333a1af8ac78553ad014509179f8075727bcb6e87f6c1e75b36c394bc6d` |
| `staged/collaboration_package_20261021_1145/OUTREACH_EMAIL_DRAFT.md` | 6841 | `3e1f1f18f22aef0c4e60a73c39a843a3006df8da2aadfc1e821ea185f3f5f3b1` |
| `staged/prevtb_subgroup_20261021_1055/PRIOR_ART_COMPARISON.md` | 8948 | `b9066a6d7cc9a529cac42c2a1eb1b8c9f11ea7b94a67f4c4ecd82528a2a43201` |
| `tables/cohort_identity_and_reanalysis.json` | 6573 | `9aca2749fb529a3bd5d4e3fc8738551ad4102030fa7fcb56954988cdaea08c56` |
