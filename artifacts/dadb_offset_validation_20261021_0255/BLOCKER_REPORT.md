# Blocker report — why the DA/DB offset validation was stopped before scoring

**Stopped at Phase 0, precondition 6. No DA/DB score was computed.**

Phase 0 instruction 6, verbatim:

> Verify that the correction parameters can be applied to DA/DB using information
> legitimately available at inference. If the method requires DA/DB labels, healthy-case
> labels, or fitting offsets on DA/DB, do not apply it as a validation. Stop and explain
> why. If DA/DB scanner IDs cannot be mapped to the source groups for which offsets were
> estimated, do not guess a mapping.

Both halves of that instruction are triggered. Each independently blocks the evaluation.

---

## Blocker 1 — the frozen offsets cannot be transferred to DA/DB

The frozen parameters are three numbers keyed to **Vietnam `rad_system` ∈ {1, 2, 3}**:

```
offset[1] = +0.5137025   offset[2] = -0.8610015   offset[3] = +2.3451740
score_corrected = score - offset[rad_system]
```

DA and DB are a different pair of acquisition sources at a different institution (NITRD,
New Delhi). **There is no mapping from `DA`/`DB` to `rad_system` 1, 2 or 3**, and no
metadata anywhere in the repository asserts one. The instruction explicitly forbids
guessing one, and guessing would be meaningless anyway: an offset of +2.345 logits was
estimated from one specific Vietnamese radiography system's score distribution and carries
no information about a Delhi scanner.

So the transform cannot be applied to DA/DB with its parameters held fixed.

## Blocker 2 — the DA/DB-specific estimator requires DA/DB healthy-case labels

`NEXT_STEP.md` anticipated Blocker 1 and specified a site-local estimator instead:

> Per site, the median score of that site's non-TB cases. DA/DB has no development half, so
> use leave-one-out: centre each case by the median of the *other* non-TB cases at its site.

That estimator **requires knowing which DA/DB cases are non-TB** — exactly the
"healthy-case labels" that precondition 6 rules out. Leave-one-out protects against a case
influencing its own correction; it does **not** make the procedure label-free. The outcome
labels of 102 negatives would still determine the offsets applied to all 176 cases.

## Blocker 3 — no label-free variant escapes the precondition either

The obvious workaround is to centre by the median of **all** cases at a site, which needs
no labels. That still fails, for two reasons:

1. Precondition 6 bars *"fitting offsets on DA/DB"* in any form, not just label-dependent
   fitting.
2. It would not be the preregistered method. Substituting a different estimator after
   reading the preconditions is precisely the post-hoc protocol change Phase 0 instruction
   3 prohibits.

## The underlying problem, which is structural rather than a data gap

**A per-source additive offset is, by construction, a source-fitted parameter. There is no
frozen parameter to carry to a source it was not fitted on.**

This means the method as framed in `NEXT_STEP.md` can *never* be validated as a
frozen-parameter transfer on a new site — not on DA/DB, and not on any future cohort. The
previous run's recommendation contained this flaw and did not notice it. It proposed
leave-one-out estimation on DA/DB in the belief that was a validation; under this task's
stricter and correct reading of what "validation" requires, it is not.

Two genuinely different questions were being conflated:

| question | what it needs | status |
|---|---|---|
| **Q1.** Do these specific Vietnam offsets transfer to a new site? | a mapping from DA/DB to `rad_system` | **impossible — no such mapping exists or could** |
| **Q2.** Does the *procedure* "centre scores per source, then score" generalise to a new site? | a new site with enough cases to estimate an offset, and a protocol that permits fitting there | **answerable, but it is not a frozen-parameter validation and precondition 6 excludes it here** |

The previous run's recommendation was written as if DA/DB could answer Q1. It cannot. It
could, under a differently worded protocol, contribute to Q2.

## Two further gaps, recorded although Blockers 1–3 already decide the matter

**The baseline score does not exist for DA/DB in the required representation.** The Vietnam
offset result was measured on focal-pooling scores (12×12 regions, max aggregation). DA/DB
has only global 768-d embeddings; `artifacts/p2/regional/` contains no DA/DB cache. A
comparable baseline would require a new regional embedding pass over 176 images. That is
cheap and permitted in principle, but it is not "using existing predictions/embeddings".

**The DA/DB embeddings carry no case IDs.** `radino_dadb_*_v1.npz` stores only `emb`
(176, 768) and `n_sanity_fail`. Alignment to labels is positional, through
`sorted(SRC.rglob("*"))`. This run verified that the reconstruction is deterministic and
reproduces the published per-site counts exactly (DA 72/22, DB 104/52, pooled 176/74), so
it is **resolvable** — but Phase 1 instruction 2 asks for scores aligned *by case ID*, and
the stored artifact does not support that without the reconstruction step. Any future DA/DB
work should write `image_id` into the npz.

Separately, `radino_dadb_lungs_only_v1.npz` records **`n_sanity_fail = 1`**, yet
`dadb_third_site.json` reports n = 176 for the lungs-only variant. The case that failed the
segmentation sanity check **was scored rather than excluded**. That is a pre-existing
inconsistency in the published artifact, not something this run introduced, and it would
need an explicit exclusion rule before any future DA/DB evaluation.

## What would need to change to make this answerable

Stated as a request for clarification, not as a proposed workaround to be adopted
unilaterally:

1. **Decide which question is being asked.** If it is Q1, DA/DB is the wrong cohort and no
   cohort can answer it; the question itself should be withdrawn. If it is Q2, say so
   explicitly and authorise site-local offset estimation on the evaluation cohort.
2. **If Q2: pre-specify a label-free estimator**, for example the median of *all* scores at
   a site, and state plainly that this is a *procedure* validation in which parameters are
   fitted on the evaluation cohort without using its outcome labels. Fix the acceptance
   rule before looking, as `NEXT_STEP.md` already does (+0.02, CI excludes zero, no site
   down more than 0.02).
3. **Authorise the regional embedding pass** for DA/DB, or accept that the comparison would
   run on the global representation and therefore would not match the Vietnam result.
4. **Fix the exclusion rule** for the one segmentation sanity failure.
5. **Note the power ceiling before committing.** With two sites, 176 cases and 74 positives,
   DA/DB is smaller than the Vietnam REPORT half that already failed to resolve this effect
   (1,356 cases, 141 positives, CI [−0.0030, +0.0436]). A null on DA/DB would be
   uninformative. This should be weighed before spending the cohort's one clean read.

Point 5 is the one worth pausing on. DA/DB's value is that nothing has used it for
selection. Spending it on a question it is statistically unable to answer would waste that.
