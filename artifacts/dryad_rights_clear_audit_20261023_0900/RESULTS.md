# Results

## No concept score was computed. This is a pending-results template, not a result.

**Status: PREPARED AND RUNNABLE, BLOCKED ON HOST MEMORY.** The two blockers that stopped
the previous run are resolved — the data licence is CC0-1.0 and the concept mapping is
pinned to 9 exact vocabulary entries. What stops execution now is a measured machine
constraint, recorded rather than worked around.

---

## What actually ran

| stage | status |
|---|---|
| Phase 0 repository and package verification | **executed locally** |
| Phase 1 data-rights assessment | **executed** — three official sources retrieved |
| Phase 2 CLEAR feasibility and concept mapping | **executed** — vocabulary downloaded, 368,294 entries, 9 concepts pinned to exact indices |
| Phase 3 concept inference on TUNE | **PREPARED, NOT EXECUTED** — see the blocker |
| Phase 4 supplementary probe | **deliberately not re-run** — see below |
| Phase 5 package | **executed** — 20 synthetic tests pass |

## The blocker

CLEAR needs a 603 MB checkpoint plus a slice of a 1,131 MB concept-embedding tensor.
Measured host memory at run time: **0.77–0.84 GB available of 16.39 GB, 95% used, with
2.63 GB resident in MemCompression** — the machine is already swapping. VRAM (5.3 GB free)
and disk (185 GB) are ample; **host RAM is the binding constraint.**

Running anyway would risk a swap-thrashing failure partway through a 1,352-image encode.
`src/run_concept_audit.py` is complete and dry-validated; it needs only `--run-inference`
on a machine with roughly 4 GB of headroom, or a Kaggle session where this is trivial.

## Pending result tables

These are written by the run and are **empty until it executes**:

| table | contents |
|---|---|
| `tables/concept_associations.csv` | per concept x group: n, Spearman rho, bootstrap 95% CI, raw p, BH-adjusted p |
| `tables/concept_incremental_r2.csv` | R2 of classifier score on clinical/site variables, with and without each concept |

Groups: negatives (all), negatives with past TB, negatives without past TB, positives
(all). **Aggregate only — no per-case concept score is written to the bundle.**

## What has already been measured, and is not re-run

From `artifacts/kaggle_concept_raydino_20261021_1830/`, Vietnam TUNE, score-only:

| group | n | median classifier score |
|---|---:|---:|
| negative, no past TB | 796 | -0.3475 |
| negative, past TB | 417 | +2.0509 |
| positive, no past TB | 90 | +3.1505 |
| positive, past TB | 49 | +3.0858 |

Negatives: difference in medians **+2.3984 [1.9685, 3.1057]**, separation **AUROC 0.6957**.
Among positives the same separation is **0.4973 — chance**.

A past_tb linear probe on frozen features was also already run (TUNE balanced accuracy
**0.9737** against 0.3381 under label permutation). **Phase 4 was therefore not re-run.**
Repeating it would add length, not evidence, and the protocol says not to. Its
interpretation stands unchanged: **predicting clinical history does not identify fibrosis
and does not show the TB classifier uses it.**

## No metric changed

**The reported Vietnam screening AUROC remains 0.7242.** Nothing in this run retrained,
rescored, recalibrated or re-thresholded anything. A resolved licence, a runnable
notebook and a pinned concept mapping are **not** a performance improvement, and are not
described as one.
