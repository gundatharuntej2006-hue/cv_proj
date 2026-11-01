# Final research-document consistency pass

**Run:** `artifacts/docfix_denominators_20261021_1700/` · 2026-10-21, 18:01 → 18:25
**HEAD `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, unchanged. `git status --short`
identical before and after** (all three files were already shown modified by earlier
documentation tasks, so none gained a new entry).

**Documentation only.** No training, scoring, rescoring, tuning or inference-code change.
**173 protected artifacts — model code, scripts, checkpoints, embeddings, predictions and
experiment manifests — verified unchanged by SHA-256.** `internal_test` untouched. No
commit, no push.

---

## 1. Were the three inconsistencies resolved? **Yes, all three.**

### Issue 1 — cohort-specific "consecutive" wording · RESOLVED

| file | before | after |
|---|---|---|
| `docs/p2_vietnam_result.md` | "**A consecutive screening cohort** is overwhelmingly the second case and then some: its Xpert-negatives include old healed tuberculosis, COPD…" | "**A screening cohort** is overwhelmingly the second case and then some: …" |
| `report/paper_draft.md` | "**A consecutive screening cohort** is the second case and more: its Xpert-negatives include healed tuberculosis, COPD…" | "**A screening cohort** is the second case and more: …" |

One word removed in each. **The substantive point about difficult negatives — healed TB,
COPD, other chronic disease, screen-detected positives — is preserved verbatim.**

Scan 1 now returns **0 cohort-specific "consecutive" claims** across all of `report/` and
`docs/`.

### Issue 2 — the 0.6744 / 0.6743 baseline · RESOLVED, after verifying provenance

**I did not take the prompt's word for it.** Checking the artifacts:

| source | value | protocol |
|---|---|---|
| **`artifacts/p2/dryrun/hard_negatives.json`** — `lungs_only` × `healthy_only` | **0.6743** [0.6443, 0.7039] | **5 seeds (17/29/43/61/79)**, n_train 3,241, TBX11K dev 0.9569 |
| `artifacts/p2/dryrun/learning_curve.json`, fraction 1.0 | 0.6742 | **3 seeds only** (17/29/43) |
| `scripts/p2_33_mil_final.py:127` | 0.6744 | a **hardcoded `print` string**, not a computed result |

**0.6744 matches no artifact.** It traces only to that console string. `hard_negatives.json`
is authoritative because it uses the five-seed protocol every other headline figure in the
project uses. Corrected to **0.6743** in the two places it appeared in
`docs/p2_focal_pooling_final.md`, each now carrying its source artifact.

### Issue 3 — mixed denominators · RESOLVED

Verified partitions:

| figure | configuration | evaluation set | n | TB | artifact |
|---|---|---|---|---|---|
| 0.6151 | full image, all negatives, global | **full cohort** | 2,708 | 280 | `vietnam_eval.json`, variant `full`, scope `all` |
| 0.6743 | lungs-only, healthy-only, global | **full cohort** | 2,708 | 280 | `hard_negatives.json` |
| **0.7242** | lungs-only, healthy-only, **regional 12×12 max** | **REPORT half** | **1,356** | **141** | `mil_final.json` |
| 0.7862 (Lunit) | commercial comparator | REPORT half | 1,356 | 141 | `mil_final.json` |
| 0.7827 (Lunit) | commercial comparator | full cohort | 2,708 | 280 | `vietnam_eval.json` |

**Two things changed down the original table, not one** — the configuration *and* the
evaluation set. The "all negatives" composition behind 0.6151 was confirmed by reading
`scripts/p2_28_vietnam_eval.py:67`, which selects `c01[c01.split == "train"]` with no
category filter.

Rewritten in both documents to state each figure with its own evaluation set, with an
explicit statement that **0.6151 → 0.7242 is not a like-for-like before/after and must not
be quoted as one improvement**, and that **a global-pooling score on the REPORT half alone
was never computed** — so no missing value was invented. Both historical values are
preserved.

**A further error found while checking the arithmetic.** The claim *"the gap to a certified
commercial product narrowed from 0.171 to 0.062"* was a **cross-denominator subtraction**:
0.7862 − 0.6151 = 0.1711 pairs a REPORT-half comparator with a full-cohort model. The
same-denominator values are **0.168** on the full cohort (0.7827 − 0.6151) and **0.062** on
the REPORT half (0.7862 − 0.7242). Corrected, with the explicit note that the narrowing
from 0.168 to 0.062 conflates the configuration change with the change of evaluation set.

**`Vietnam REPORT AUROC = 0.7242` is unchanged**, as required.

## 2. Files changed — 3

| path | sha256 before → after |
|---|---|
| `docs/p2_vietnam_result.md` | `6f8d0601b17d…` → `78a27031e315…` |
| `docs/p2_focal_pooling_final.md` | `bfe1b2ff3b84…` → `75745c7002f7…` |
| `report/paper_draft.md` | `23696132b41c…` → `21ea696d01f0…` |

Unified diff: `CHANGES.diff` (116 lines). Row-by-row log: `tables/change_log.csv`
(12 rows). Backups: `backups/`. **No file outside the three authorised paths was edited.**

## 3. Scan results

`scripts/scan.py`, output `tables/scan_results.json`.

| scan | result |
|---|---|
| 1 — cohort-specific "consecutive" enrolment claims | **0** |
| 2 — Codlin citation + article DOI + Dryad DOI in cohort-describing documents | **6 / 6 pass** |
| 3 — baseline-value occurrences | 4 × **0.6743** (authoritative), 1 × 0.6744 remaining **out of scope** (§4) |
| 4 — sentences mixing full-cohort and REPORT-half figures | 4, **all four now name their denominators** — they are the corrective text itself |
| 5 — overclaim patterns (5 checked) | 2 hits, **both false positives** (§5) |
| protected artifacts | **173 checked, 0 changed** |

**Production test suite:** `228 passed, 1 skipped` in 166 s. No GPU work, no model run.

## 4. Unresolved — 2, both outside the authorised paths

| location | issue | recommendation |
|---|---|---|
| `docs/p2_focal_pooling.md:41` | `\| production global model \| 0.6744 \|` | This is the **superseded preliminary 6×6 document**, which `p2_focal_pooling_final.md` replaces. Correct to 0.6743, or add a superseded banner. **Needs authorisation** |
| `scripts/p2_33_mil_final.py:127` | `print("  previous best global model on the full cohort: 0.6744")` | **Left deliberately.** It is a console string inside a completed experiment script that produced a committed result, and the script is inside the protected-artifact set. Editing it would alter a protected file for no scientific gain |

## 5. The two overclaim hits are false positives

Both matched the phrase inside a **negation**:

- `report/paper_draft.md`: *"It is **not** the case that regional pooling improves every
  cohort: it helps at three sites and is neutral at the fourth."*
- `report/slides/p2_slides_outline.md`: *"…regional pooling gains +0.072 on DB and nothing
  on DA, **so do not say** it improves every cohort."*

No action taken, none needed.

## 6. Did any model scores or model-related artifacts change? **No.**

173 protected artifacts — every `.npz` embedding, `.csv` prediction file, `.json`
experiment manifest under `artifacts/p2/`, and every file in `src/` and `scripts/` —
hash-verified identical. No metric was altered except the single demonstrable transcription
error in Issue 2, corrected against its authoritative artifact. `internal_test` was not
opened.

**Nothing in this task improved model performance, and it is not described as having done
so.** Every claim of clinical validation, universal pooling benefit and architectural
novelty remains absent or explicitly negated.

## 7. The single most important remaining scientific blocker

**No second screening cohort.** Every model intervention now has a measured effect with an
interval, and the curated-to-screening gap persists under all of them. The project is
precision-limited: a paired comparison between two of our own arms has a standard error of
about 0.004, while estimating what either arm is actually worth has a standard error of
about 0.019. We can tell our variants apart and cannot say how good any of them is.

The prior-art review sharpened this further: the one cohort we have is a **published
benchmark** (Codlin et al. 2025), whose previous-TB subgroup analysis for our commercial
comparator is already in print. So it cannot serve as independent validation for the
questions we most want to answer.

**Nothing in the documentation, and no further model change, resolves this. It requires
data** — roughly 2,500–3,500 analysable participants with universal microbiological
confirmation and previous-TB history recorded, per
`artifacts/prevtb_subgroup_20261021_1055/NEXT_STUDY_PROTOCOL.md`.
