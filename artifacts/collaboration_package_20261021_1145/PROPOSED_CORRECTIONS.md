# Proposed corrections to `prevtb_subgroup_20261021_1055`

**No file in that run directory was modified.** These are patches for review. All 50
numeric and methodological checks on that run passed (`tables/claim_verification.json`),
so none of the corrections below is a factual error in a number — they are additions of
newly verified prior art and one qualification of a statistical claim.

---

## Correction 1 — record the four supplement retrieval attempts

**File:** `PRIOR_ART_COMPARISON.md`, §1, after the "Discrimination:" paragraph.
**Reason:** the earlier run recorded the supplement as not retrieved without saying what
was tried. Four attempts have now been made and documented, which changes an unexplained
gap into an evidenced one.

```diff
 **Discrimination:** AUROC ≥ 0.85 overall for both systems. Stratified AUROCs are in
 Supplementary Table 2, **which was not retrieved in this run**, so their values and
 intervals are **UNVERIFIED here** and are not quoted.
+
+**Four retrieval attempts, 2026-10-21** (full log:
+`collaboration_package_20261021_1145/LITERATURE_VERIFICATION_LOG.md` §1): the PMC and OUP
+full texts both opened successfully, but the supplement
+(`ciac679_supplementary_data.docx`) could not be read — PMC serves a
+"Preparing to download…" interstitial for the binary, and the OUP CDN link requires
+signed query parameters absent from the base URL. The values remain **UNVERIFIED**.
```

## Correction 2 — add the critical unretrieved lead

**File:** `PRIOR_ART_COMPARISON.md`, §4 table.
**Reason:** a specific publication has been identified that may already report per-product
prior-TB AUROCs. The earlier run recorded this as a general "unknown"; it is now a named,
locatable paper, which is materially more useful and more constraining.

```diff
 | whether any published work reports a between-subgroup AUROC *difference with an interval* | **UNKNOWN** — not searched systematically |
+| **Qin et al., *Lancet Digital Health*, 2024** — the external CAD validation of the South African National TB Prevalence Survey, cited as ref. 17 of Qin et al. *Sci Rep* 2025;15:22540 | **NOT RETRIEVED — the single most important gap.** The *Sci Rep* paper defers CAD subgroup results to it and states most CAD products showed "significantly worse performance in … individuals with prior TB". It is the publication most likely to already contain per-product prior-TB AUROCs. **Retrieve before any model-specificity framing.** |
+| Qin ZZ et al., five-algorithm evaluation, Bangladesh, arXiv 2006.05509 (2020, **preprint**) | **NOT RETRIEVED.** A search snippet indicates it reported performance differing by prior-TB history *across algorithms*. If accurate, algorithm-specific differences were described as early as 2020. |
+| Qin ZZ et al., *Sci Rep* 2025;15:22540, DOI 10.1038/s41598-025-06164-w | **RETRIEVED AND READ.** 774 CXRs from the SA prevalence survey, 12 CAD products, 11 radiologists, 197 (26%) with prior TB. Reports overall AUCs only, not prior-TB-stratified per product. **Verified and important: radiologists degrade too** — every reader group showed significantly lower specificity in prior-TB participants. |
```

## Correction 3 — qualify the multiplicity claim

**File:** `PREVIOUS_TB_SUBGROUP_REPORT.md`, §6, "Multiplicity and selection".
**Reason:** the existing wording is defensible but states more than a Bonferroni check
delivers. The brief asked specifically that the adjustment be confirmed to match the
number and nature of the comparisons; it does not, and should be qualified rather than
dropped.

```diff
-The effect is large enough to survive a Bonferroni correction across five
-variables — the bootstrap p is below the 1/10,000 resolution of the procedure — but the
-point stands that this is a hypothesis generated and tested on the same data.
+The effect is large enough to survive a Bonferroni correction across the five screened
+variables: the bootstrap p sits at the **1/10,000 resolution floor** of the percentile
+procedure, so even at α/5 it remains below 0.0005. **That is a rough check, not a
+multiplicity treatment.** It corrects for the *selection of the variable*, not for the
+six contrasts subsequently reported from it (two subgroup AUROCs per model, two
+between-subgroup differences, the difference in differences, and the negative-only
+separability), and **no selective-inference method was applied**, so the intervals are
+not selection-corrected. The point stands that this is a hypothesis generated and tested
+on the same data.
```

## Correction 4 — state that the effect is not AI-specific

**File:** `PREVIOUS_TB_SUBGROUP_REPORT.md`, §6, after "Plausible but not demonstrated".
**Reason:** newly verified and directly relevant to how the finding should be framed.

```diff
 **Plausible but not demonstrated.** Residual radiographic changes from treated disease —
 fibrosis, cavities, volume loss, calcification — resemble active TB, and a model trained on
 `active_tb` plus `latent_tb` against healthy-only negatives has no way to learn the
 distinction. **This experiment does not test that mechanism.**
+
+**And the effect is not specific to AI.** Qin et al. (*Sci Rep* 2025;15:22540) report
+that every group of human readers in a multi-country comparison showed significantly
+lower specificity in participants with prior TB. Whatever is happening is at least partly
+a property of the images, not of any particular model.
```

## Correction 5 — align the recommended cohort size with its endpoint

**File:** `NEXT_STUDY_PROTOCOL.md`, §5.
**Reason:** two power scripts exist in the repository sizing **different** endpoints, and
the risk of someone combining them is real. The protocol's own numbers all reproduce
exactly; this is a signposting fix, not a numerical one.

```diff
 **This supersedes the "2,500–4,000" figure from the earlier run**, which was computed for
 a different endpoint (precision of a single subgroup AUROC). The ranges happen to overlap;
 the derivation is different and the one above is the one that applies here.
+
+**Two power scripts exist in this repository and they must not be combined:**
+`prevtb_subgroup_20261021_1055/scripts/power_for_subgroup_study.py` sizes the
+**between-subgroup AUROC difference** and is what this protocol rests on;
+`doc_finalisation_20261021_0930/scripts/power_planning.py` sizes a **different**
+endpoint — the paired AUROC difference between two model arms on the whole cohort, and
+the precision of a single-arm AUROC. Both were independently reproduced from their inputs
+on 2026-10-21 (`collaboration_package_20261021_1145/tables/claim_verification.json`).
```

---

## Not corrected, and why

| candidate | decision |
|---|---|
| every numeric value in both documents | **all reproduce exactly** — 50/50 checks pass. No change |
| the shared-positive bootstrap design | correct as written and asserted by test. No change |
| the "0.7811 even without previous TB" caveat | present and correctly stated. No change |
| the Phase 1 / Phase 2 score separation | correctly kept apart in both documents. No change |
| `COLLABORATION_OUTREACH_DRAFT.md` in the earlier run | superseded by `OUTREACH_EMAIL_DRAFT.md` here, which adds the unretrieved-supplement disclosure, the cohort-size shortfall, and reframes the ask as feedback first. **The earlier draft is left in place as the record of what was written then** |

## How to apply

These are proposals. If accepted, apply them to the files in
`artifacts/prevtb_subgroup_20261021_1055/` and regenerate that run's `FILE_MANIFEST.md`,
since the hashes recorded there will change.
