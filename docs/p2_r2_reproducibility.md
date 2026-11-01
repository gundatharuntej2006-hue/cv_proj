# The R2 "reproduces to 1e-4" claim is two claims, and only one of them holds

**Owner:** Person 2 · **Date:** 2026-10-14 · **Affects:** R2 contract, D14, Person 4's P4-09
gate · **Status:** measured; needs a one-line contract amendment

## Summary

The R2 contract says `p2_cls.api.classify` reproduces the exported C08 scores to **1e-4**.
Measured, that sentence turns out to contain two different claims:

| claim | what it compares | worst abs diff | passes 1e-4 |
|---|---|---|---|
| **A. head reproducibility** | the registered head re-run on the *same stored embedding* vs the exported `p_tb_raw` | **1.9 × 10⁻⁷** | **yes**, by three orders of magnitude |
| **B. end-to-end reproducibility** | the *image* through the runtime path vs the exported `p_tb_raw` | **5.7 × 10⁻⁴** (median 8.3 × 10⁻⁵, p95 3.4 × 10⁻⁴) | **no** |

Claim A is what `registry.reproduces_scores` checks and it passes comfortably — the head is
a linear layer, so re-running it is the same matmul on the same input.

Claim B fails, and **1e-4 was never achievable for it by construction**. C07 was produced by
`embed.Embedder` running the backbone in **float16 on the GPU** and stored as float16;
the runtime path in `api.classify` runs **float32 on CPU**. float16 carries roughly three
decimal digits, so a disagreement of a few times 10⁻⁴ in the output probability is the
expected floor, not a bug. Nothing is wrong with either path; the stated tolerance was
written without the dtype in mind.

Measured with `scripts/p2_13_verify_r2.py --n 48`, artefact
`artifacts/p2/dryrun/r2_verification.json`.

## Does it matter clinically?

This is the only question that decides whether anything needs fixing. A difference in the
probability matters if and only if it can move a case across a triage threshold.

Taking the worst-case gap (5.7 × 10⁻⁴, scaled by the calibration temperature to 1.2 × 10⁻³
on the calibrated scale), **5 of 8,400 cases (0.06%)** sit close enough to `t_lower` that the
difference could in principle flip their category. For the remaining 99.94% the two paths
agree on the decision regardless.

0.06% is small but it is not zero, and "the runtime and the reported table disagree on five
cases" is exactly the kind of thing that is embarrassing to discover during a demo rather
than now.

## Recommendation

**1. Split the contract wording.** Two sentences instead of one:

> `classify` reproduces the exported C08 `p_tb_raw` to **1e-4** when given the same C07
> embedding (the D14 head-reproducibility check). Image-to-probability agreement between the
> float16 export path and the float32 runtime path is **1e-3**, which is the float16 floor.

Both numbers are then met with margin, and P4-09's gate can check the right one.

**2. For the final run, make the two paths the same dtype.** When Person 1's C01 lands and
the final C07/C08 are exported for the chosen variant, run the embedder in float32 so the
export and the runtime agree exactly. Cost: one re-embed of **one** variant (~30 min at
float32 instead of 14 at float16) and 25 MB instead of 13 MB of storage. There is no reason
to do this for the nine variants that exist only to decide D08/D09 — they are compared
against each other, all in float16, which is a fair comparison.

**3. Do not "fix" it by loosening everything to 1e-3.** Claim A passing at 1.9 × 10⁻⁷ is a
genuine guarantee that the registry is intact and the weights on disk are the weights that
produced the table. Folding it into a loose end-to-end tolerance would throw away a sharp
check to accommodate a soft one.

## For Person 4

`p2_cls.api.classify` is **no longer a stub**: with a bundle present at
`artifacts/p2/bundle/` it loads the trained head and returns `is_stub=False`, so the P4-09
gate will now accept its numbers. `api.is_stub()` answers the question without running a
forward pass, and `TB_P2_FORCE_STUB=1` pins the old stub behaviour for tests. The same is
true of `classifier_module()` for Person 3's Grad-CAM, which now also exposes `tb_index`
because the TB column is 1 for a binary head and 2 for a three-class one.

One wiring detail that matters if you ever call the backbone yourself: the head was trained
on **L2-normalised** features, which is what `embed.Embedder.embed` writes. Feeding it a raw
`pooler_output` produces confident, plausible, wrong probabilities rather than an error.
`api._features` does the normalisation; the Grad-CAM module's forward does it too, so the
heatmap explains the number that is actually reported.
