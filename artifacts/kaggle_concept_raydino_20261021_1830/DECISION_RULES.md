# Decision rules

**Written and hashed before any encoder is loaded or any new score computed.** These rules
are not to be reinterpreted after seeing a result.

---

## Stage 1 — concept audit

**Hypothesis.** Among Xpert-negative participants, higher classifier scores are associated
with model-derived concept scores for chronic abnormality — fibrosis, scarring, volume
loss, calcification — and this association is stronger in those with a documented history
of previous TB.

**Endpoints.** Per concept: Spearman correlation with the frozen classifier score,
stratified by active-TB label and previous-TB history. Bootstrap CIs, 10,000 replicates.
**Benjamini–Hochberg across the 12 preregistered concepts**, α = 0.05.

**What would count as a credible, actionable signal — all four required:**

1. at least one chronic-abnormality concept correlates with the classifier score among
   **negatives** with a BH-adjusted interval excluding zero;
2. the same concept does **not** correlate as strongly among **positives** — otherwise
   suppressing it would suppress genuine active or recurrent disease;
3. the effect survives stratification by previous-TB history, i.e. it is not merely a
   restatement of the history variable;
4. the concept maps to an exact phrase in the official CLEAR concept bank, not an
   approximation.

**What would not count.** A score difference by `past_tb` alone. That is already measured
(TUNE: negatives with past TB median +2.05 against −0.35 without, difference +2.40
[1.97, 3.11]) and it is **not** evidence that the classifier uses any visual cue.
`past_tb` is clinical history, not a radiologist annotation.

**Stop rule.** If images or the concept bank are unavailable, or the image licence is
unresolved, **report the blocker and compute nothing.** Do not substitute a proxy.

**Expectation, stated up front:** a concept audit is diagnostic. It is **not expected to
change AUROC** and no AUROC change should be reported from it.

## Stage 2 — encoder benchmark

**Hypothesis.** A different frozen chest-radiograph representation changes downstream
TB discrimination under a matched head. **Direction is not assumed.**

**Primary endpoint.** AUROC on TBX11K **development**, matched global heads, RayDINO minus
RAD-DINO, with a paired bootstrap CI (4,000 replicates, seed 0) on identical cases.

**Minimum meaningful effect: +0.02 AUROC.** Chosen before any fit. Below that, replacing a
frozen encoder — with a larger model, a different licence and a gated download — is not
worth the operational cost.

**Decision rule — all three required for ADOPT-FOR-FURTHER-STUDY:**

1. paired ΔAUROC ≥ **+0.02**;
2. the 95% paired CI **excludes zero**;
3. **no unacceptable safety loss**: sensitivity at 70% specificity must not fall, and in
   particular must not fall in the **previous-TB subgroup**, where this project already
   knows discrimination is weakest.

Any other outcome is **REJECT** or **INCONCLUSIVE**, recorded as such. A favourable point
estimate with an interval containing zero is **not** a positive result.

**Explicitly not a win:** a gain on Vietnam TUNE alone. TUNE has been used repeatedly and
is exploratory. **Vietnam REPORT is not read at all.**

**Secondary, reported but not decisive.** Regional comparison via the declared
`pool_to_grid` harmonisation; AUPRC with prevalence; specificity at fixed sensitivity;
previous-TB subgroup metrics; extraction time, peak GPU memory, feature storage size.

**Multiplicity.** One primary endpoint, one comparison. Secondary endpoints are labelled
secondary and are not promoted.

## Stage 3 — conditional intervention

**Runs only if Stage 1 yields all four signal conditions AND Stage 2 completed validly.**
Otherwise it does not run, and the reason is recorded.

If it runs: at most **two prespecified weights including the neutral 1.0**. No grid
search. Evaluation on **development** only, paired against the frozen baseline.

**Hard limits.** The prior applies to per-region *disease scores*, never raw tokens. No
subtraction of a past-TB score from an active-TB logit. No treating concept predictions as
fibrosis ground truth. **Apical location is not a universal TB rule** — active TB occurs
outside the apices and old scarring is frequently apical, so an apical up-weight could
worsen exactly the confusion it targets.

## Rules that bind every stage

- **No causal claim** from probes, saliency, attention, concept scores or correlations.
  These are associations and hypotheses.
- **No clinical readiness claim**, and no "solution" claim from AUROC alone. Sensitivity,
  specificity, uncertainty and subgroup effects are reported together.
- **An underpowered null is inconclusive**, never evidence of no effect.
- **`internal_test` is never touched.** Not read, scored, enumerated, or named in any
  manifest or log beyond its row count.
- **No deployment recommendation** follows from any of this.
- **A local dry run is not a Kaggle run.** No Kaggle result may be claimed unless the
  notebook actually executed there and its outputs exist.
