# Documentation fix: the two remaining cohort-provenance corrections

**Run:** `artifacts/docfix_slides_focal_20261021_1415/` · 2026-10-21, 16:31 → 16:50
**HEAD `f878a6c9e9746803938463941ae1ffaa5d1fd96a`, unchanged before and after.**

**Documentation only. This task did not and cannot improve model performance.** No
training, scoring, feature extraction or threshold fitting. No `internal_test` access. No
model code, checkpoint, embedding, prediction file, experiment manifest or result was
touched — **173 protected artifacts verified unchanged by SHA-256**. No commit, no push.

---

## 1. Exact paths changed — 2

| path | sha256 before → after |
|---|---|
| `report/slides/p2_slides_outline.md` | `68cc5a5808c3…` → `f04093e27b50…` |
| `docs/p2_focal_pooling_final.md` | `034436ef8996…` → `ce1f0908285b…` |

Backups of both pre-edit files: `backups/`. Unified diff: `CHANGES.diff` (79 lines).
Row-by-row log: `tables/change_log.csv` (11 rows). Hash table:
`tables/hashes_before_after.csv`.

**No other tracked file was edited.** Both files contained exactly the claims the previous
run reported, so no stop condition was triggered on that count.

## 2. Before and after, every change

### `report/slides/p2_slides_outline.md`

**Slide 10 table row**
- before: `| **Vietnam** | **consecutive screening cohort** | **0.6151** |`
- after: `| **Vietnam** | **screening cohort (risk-based programmatic ACF)** | **0.6151** |`

**Slide 11 headline**
- before: *"**0.6151** on 2,708 consecutive screening attendees, Xpert Ultra on every one."*
- after: *"**0.6151** on a 2,708-participant screening test library, Xpert Ultra on every
  participant in it."*

Two corrections in one sentence: the enrolment description, and the Xpert claim — now
scoped to the test library rather than to everyone screened, which the source does not
support.

**Slide 11, new speaker note** (added): the Dryad DOI, the full Codlin et al. citation with
DOI, identification by matching title, author list and six reproduced AUROCs and
**explicitly not by patient-ID matching**, the risk-based ACF recruitment description, an
instruction not to call it an unselected consecutive cohort, and the fact that the
commercial comparator figure is that paper's published comparator rather than a benchmark
assembled here.

### `docs/p2_focal_pooling_final.md`

**Headline sentence**
- before: *"…raises performance on a consecutive TB screening cohort from **0.6151 to
  0.7242**…"*
- after: *"…raises performance on a TB screening cohort from **0.6151 to 0.7242**…"*

**New "Cohort provenance" blockquote** immediately after it, in the style already used in
`docs/p2_external_validation.md`: Dryad DOI, full citation with DOI, the
statistical-reproduction basis with the explicit statement that **participant-level
identity was not verified**, the risk-based ACF recruitment, and the note that the Lunit
figure is that paper's published comparator.

**"Relation to prior work", fourth bullet**
- before: *"Our evaluation adds a consecutive screening cohort with an Xpert reference
  standard and a commercial comparator on identical patients, which the RADAR page does
  not report. That is an evaluation difference, not an architectural one."*
- after: same, with "consecutive screening cohort" replaced by *"screening cohort — risk-based
  programmatic active case finding, not unselected consecutive enrolment —"* and this
  sentence appended: *"Note that the comparator and the previous-TB subgroup analysis on
  this cohort are themselves published (Codlin et al. 2025), so this is not an independent
  evaluation asset of ours."*

That last addition matters: the bullet was claiming the cohort as a distinguishing
evaluation strength. It is a published benchmark, and the previous-TB analysis on it is
published too.

## 3. Consistency scan results

`scripts/final_consistency_scan.py`, output `tables/final_consistency_scan.json`. The scan
separates descriptions of **this cohort** from **generic** statements about consecutive
sampling, classifying by proximity to cohort markers rather than by the word alone.

| section | result |
|---|---|
| **A** — "consecutive" near a cohort marker, needs review | **2**, both outside this task's authorised files (§4) |
| **B** — generic or corrective statements, not errors | 11, all correctly classified |
| **C** — novelty / overclaim patterns (6 patterns) | **1, and it is a false positive** (§5) |
| **D** — Codlin citation + DOI present in cohort-describing files | **6 / 6 pass**, including both files edited here |
| **E** — protected artifacts | **173 checked, 0 changed** |

No hit for: the previous-TB effect framed as novel; the effect generalised to all CAD
products; clinical readiness; independent final validation of an exposed cohort; a causal
scanner claim.

## 4. Remaining hits requiring human review — 2

Both are the same sentence pattern, in files **this task was not authorised to edit**. Per
the stop conditions they are listed rather than changed.

| file | text |
|---|---|
| `docs/p2_vietnam_result.md` | *"**A consecutive screening cohort** is overwhelmingly the second case and then some: its Xpert-negatives include old healed tuberculosis, COPD and other chronic disease…"* |
| `report/paper_draft.md` | *"**A consecutive screening cohort** is the second case and more: its Xpert-negatives include healed tuberculosis, COPD…"* |

**Judgement, offered not acted on:** each reads as a general claim about what consecutive
screening cohorts contain, used to explain the negative-class spectrum — but in context it
is being applied to this cohort. The point about the negative spectrum holds regardless of
enrolment design, so the minimal fix is to drop the word: *"A screening cohort is…"*.
**Ask before editing** — these files are outside scope.

## 5. The one novelty hit, and why it is a false positive

The scan flagged `docs/p2_focal_pooling_final.md` for "architectural novelty". The matched
text is a **negation**:

> "What this document reports is a *controlled measurement* of the aggregation choice under
> one pipeline, **not a new method**."

The regex matched "new method" inside the disclaimer. **No action taken, and none needed.**
The file's existing non-novelty statement — *"This architecture is not novel"* — stands.

## 6. Pre-existing issues in a target file, left alone

Two items in `docs/p2_focal_pooling_final.md` are out of scope here and were already
recorded elsewhere; neither was touched:

- The ladder table quotes the global baseline as **0.6744**, while the committed artifact
  (`hard_negatives.json`, five seeds) gives **0.6743**. Already Finding 2 of
  `artifacts/doc_finalisation_20261021_0930/REPORT_CONSISTENCY_CHECK.md`.
- The headline compares **0.6151** (full cohort) with **0.7242** (REPORT half) — different
  denominators. Documented in `docs/p2_vietnam_result.md` and `report/paper_draft.md` §5.0.

## 7. Git state and verification

| | |
|---|---|
| HEAD before / after | `f878a6c9e9746803938463941ae1ffaa5d1fd96a` / identical |
| `git status --short` diff | **one line added**: `M docs/p2_focal_pooling_final.md`. The slides file was already shown modified by an earlier task, so it gained no new entry |
| protected artifacts | **173 checked, 0 changed** |
| commits / pushes | **none** |
| checks run | documentation scan only; **no GPU suite, no training, no scoring** |

Before/after `git status` and HEAD are captured verbatim in
`tables/git_status_before.txt`, `tables/git_status_after.txt`,
`tables/git_head_before.txt`, `tables/git_head_after.txt`.

## 8. Note on the scan script

It needed two corrections while being written: the Windows console is cp1252 and crashed
on `≥` and an en dash, and an earlier in-place patch of the print statements left unmatched
parentheses. The script was rewritten cleanly with an encoding-safe output helper. Neither
problem affected any document — both were in the reporting path.
