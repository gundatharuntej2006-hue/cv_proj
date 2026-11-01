# Final report — Dryad rights and CLEAR concept audit

**Run:** `artifacts/dryad_rights_clear_audit_20261023_0900/` · 2026-10-23
**HEAD before and after:** `f878a6c9e9746803938463941ae1ffaa5d1fd96a` — unchanged
**No concept score was computed. The reported Vietnam screening AUROC remains 0.7242.**

---

## 1. What was verified from official sources

**The Dryad blocker was wrong and is withdrawn.** Three primary sources, all retrieved
2026-10-23:

| source | finding |
|---|---|
| Dryad reuse guidance, `v3.datadryad.org/help/guides/reuse` | *"All datasets on Dryad are published under the CC0 (Creative Commons Zero) Public Domain Dedication."* Explicitly permits reuse, modification, sharing and **redistribution**. No exceptions listed |
| Dataset landing page | **Displays no licence** beyond a generic footer link. This absence is what caused the original blocker |
| **Dataset API record** | **`"license": "https://spdx.org/licenses/CC0-1.0.html"`** — dataset-specific, machine-readable, version 5, published 2025-03-14 |

The earlier caution rested on a README omitting a licence line. **A README omitting a
licence is not a licence restriction**; the deposit's licence lives in repository
metadata, and it says CC0-1.0.

Also verified: **CLEAR is Apache-2.0** for code and weights, pinned to HF revision
`142f169c8b006b0f28cf55c6a237c3003db5ed8f`, DINOv2 ViT-B/14 with registers at 448 px.
**RayDINO was not touched** — gated, FAIR Noncommercial Research License, out of scope.

## 2. Is the licensing now clear enough?

**Status: CLEAR FOR THE PROPOSED PUBLIC-DATA RESEARCH USE.** High confidence on the
copyright question: an archive-wide policy and a dataset-specific metadata field agree,
and no contradictory notice exists on the landing page, in the README, or in the metadata.

**Stated cautiously, because three things remain true:**

1. **CC0 waives copyright, not human-subjects obligations.** These are de-identified
   radiographs of real patients. The licence is not an ethics approval and not a finding
   that re-identification risk is zero.
2. **CC0 on the data says nothing about model weights.** RayDINO stays gated and
   non-commercial. Separate instruments, separate answers.
3. **Permitted is not the same as prudent.** CC0 allows redistribution; the recommendation
   is still a **private** Kaggle dataset.

## 3. What remains blocked

| component | status |
|---|---|
| Vietnam image rights | **RESOLVED — CC0-1.0** |
| CLEAR code and weights | **RESOLVED — Apache-2.0, revision pinned** |
| Concept-to-output mapping | **RESOLVED — 9 of 12 concepts pinned to exact vocabulary indices** |
| **Phase 3 execution** | **BLOCKED on host RAM** — 0.77–0.84 GB free of 16.39, 95% used, 2.63 GB in MemCompression. CLEAR needs a 603 MB checkpoint plus a slice of a 1,131 MB tensor |
| RayDINO benchmark | **OUT OF SCOPE** — gated weights, unresolved redistribution |
| cavity, nodule, reticular opacity | **EXCLUDED** — no exact vocabulary entry; not approximated |

## 4. Did concept inference run? **No.**

It did not run, on any split. The script is complete and dry-validated: it loads the
1,352-patient TUNE cohort, confirms 0 REPORT patients and no `internal_test` reference,
and resolves 9 concepts — then stops, because `--run-inference` was not passed.

**No concept score exists.** Pending tables are specified in `RESULTS.md`.

## 5. Aggregate results

**None from concept inference.** The only measured quantities are Phase 2's vocabulary
coverage, which is a property of CLEAR rather than a result about our model:

- CLEAR's vocabulary is **368,294 free-text report sentences**, not a curated finding set.
- **9 of 12** preregistered concepts have an exact single entry; **3 do not and are
  excluded**, including **cavity**, which matters clinically in TB.
- Negation rates among matching sentences are substantial — **28.6%** for consolidation,
  **24.5%** for focal opacity — which is why the exact-entry rule was adopted rather than
  aggregating.

Multiple-testing correction (Benjamini–Hochberg across the 9 concepts) and bootstrap
intervals are implemented and unit-tested, but have nothing to correct yet.

**Phase 4 was deliberately not re-run.** The `past_tb` probe already exists from an earlier
run (TUNE balanced accuracy **0.9737** against 0.3381 under permutation). Repeating it
would add length, not evidence. Its reading is unchanged: predicting clinical history does
not identify fibrosis and does not show the TB classifier uses it.

## 6. Tests

**20 synthetic tests, all passing** (8.5 s), plus `py_compile` clean.

**One test failure during development, and it was a real finding.** A test injecting a
REPORT patient into the score file did not raise. Investigating showed the loader selects
by allow-list (`pid in tune`), so contamination is prevented *by construction* and the
residual intersection check is dead code that can never fire — giving false assurance. The
test was rewritten to exercise guards that genuinely can fire: a **missing** TUNE patient,
a **duplicate** patient, and an **overlapping split manifest**. All three now fail loudly,
and the structural-impossibility argument is itself asserted. **The test was wrong; the
guard's weakness it exposed was real.**

## 7. What changed

| | |
|---|---|
| model predictions, checkpoints, caches | **unchanged** |
| tracked source or documents | **unchanged** |
| previous run directories | **unchanged** except `kaggle_concept_raydino_20261021_1830/FINAL_RUN_REPORT.md`, which differs from that package's own manifest because it was edited after the manifest was generated, in the run that created it. Nothing in this run touched it |
| `git status --short` | one added line for this run directory |
| commits, pushes, uploads, emails | **none** |
| Vietnam REPORT | **never opened** |
| TBX11K `internal_test` | **never opened** |

One file was written outside the run directory: `D:\hf-cache\clear_vocab\mimic_concepts.csv`
(22.2 MB), the CLEAR vocabulary, placed in the shared HF cache per the project's
convention rather than in the bundle. The bundle stays code-only — asserted by test.

## 8. Single best next action

**Run `--run-inference` on Kaggle.** Everything else is resolved: the licence is CC0, the
assets are Apache-2.0 and pinned, the mapping is fixed to 9 exact indices, the guards are
tested, and the notebook is written. The only obstacle is 3 GB of RAM this machine does
not currently have, and Kaggle has it for free.

Expect a diagnostic, not an improvement. The most likely outcome is a handful of
chronic-abnormality concepts correlating with the classifier score among negatives — which
would be **consistent with** the known previous-TB effect and would still not show the
classifier uses any visual cue. Per `DECISION_RULES.md` in the prior package, a credible
actionable signal needs all four conditions, and a score-by-history association alone does
not qualify.
