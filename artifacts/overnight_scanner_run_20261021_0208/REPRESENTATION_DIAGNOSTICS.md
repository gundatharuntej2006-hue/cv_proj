# Task 4 — Did the projection remove scanner information, and did TB survive?

**TUNE only. The scanner probe is 5-fold cross-validated and never scores rows used to fit
it. REPORT is not read anywhere in this document.**

Tables: `tables/representation_diagnostics.csv`, `tables/inlp_diagnostics.csv`,
`tables/inlp_diagnostics.json`

---

## 1. A correction to this task's own first attempt

The first run swept the subspace rank over 2, 3, 8, 16, 32 and 64 and produced
**byte-identical results for every rank above 2**. That is a bug, not a finding: the
subspace comes from a 3-class linear layer, so `torch.linalg.svd` on its 3 × 768 weight
matrix returns only three right-singular vectors. Asking for rank 64 silently returned the
same three rows.

The degenerate sweep is kept in `representation_diagnostics.csv` with this note attached,
rather than deleted. Removing more scanner information requires **iterative nullspace
projection**: fit a scanner classifier on the current representation, project out its
subspace, refit on what remains, repeat. Each round removes up to two further directions.
That is what the corrected analysis below does.

## 2. Chance level, established by permutation

The probe was re-run on **20 permutations of the scanner labels**, seed 17, with the same
5-fold cross-validation: **balanced accuracy 0.3369** (sd 0.0130, max 0.3615). Three
classes, so chance is a third, as expected.

## 3. The result

Frozen TB head (trained on TBX11K, never on Vietnam). Probe reads the mean-pooled 768-d
token vector per image. Both quantities measured on the **same** Vietnam TUNE cases at
every step.

| INLP round | directions removed | scanner balanced accuracy | TB AUROC | TB 95% CI |
|---|---|---|---|---|
| 0 (baseline) | 0 | **0.9896** | 0.7237 | [0.6852, 0.7614] |
| 1 ≈ preregistered rank 3 | 3 | 0.9763 | 0.7233 | [0.6851, 0.7610] |
| 2 | 6 | 0.9709 | 0.7231 | [0.6847, 0.7610] |
| 3 | 9 | 0.9640 | 0.7236 | [0.6853, 0.7614] |
| 4 | 12 | 0.9604 | 0.7234 | [0.6851, 0.7613] |
| 6 | 18 | 0.9461 | 0.7239 | [0.6855, 0.7617] |
| 8 | 24 | 0.9428 | 0.7242 | [0.6860, 0.7619] |
| 12 | 35 | 0.9303 | 0.7237 | [0.6849, 0.7613] |
| 16 | **44** | **0.9118** | **0.7242** | [0.6860, 0.7615] |

Chance for the probe is 0.3369.

## 4. Two findings, both clear

### 4a. Scanner identity is far too redundantly encoded to project away

Removing **44 of 768 dimensions** — the full linear scanner signal found over sixteen
successive rounds — drops scanner balanced accuracy only from **0.9896 to 0.9118**. It
remains **0.575 above chance**. Each round finds fresh scanner-predictive directions in
what the previous rounds left behind.

The information is not sitting in a small nuisance subspace waiting to be deleted. It is
distributed across the representation, which is unsurprising for a frozen foundation model
trained on 883,000 radiographs from five sources: acquisition characteristics are part of
what it learned to represent.

**This means low-rank linear nuisance removal is simply the wrong instrument here**, and
the preregistered rank-3 version never had a realistic chance of working.

### 4b. Tuberculosis discrimination is completely insensitive to the removal

Across the whole sweep, TB AUROC moves between **0.7231 and 0.7242** — a total range of
0.0011, with every confidence interval essentially superimposed. Removing 44 of the most
scanner-predictive directions in the representation costs the tuberculosis classifier
nothing measurable.

## 5. What this does and does not establish

**Established.** The directions along which scanner identity is linearly decodable are
not the directions the tuberculosis head depends on. The two signals are, to this
resolution, linearly separable within the frozen representation.

**Not established, and explicitly disclaimed.**

- **Reduced scanner prediction does not prove a clinical shortcut was removed.** Scanner
  predictability is a property of the representation, not of the decision. A model can
  score correctly for bad reasons that no scanner probe would reveal.
- **Nor does it prove causal fairness improved.** Nothing here is a causal analysis. The
  observed scanner disparity in referral rates is untouched by any of these projections —
  see `CALIBRATION_AND_ERROR_ANALYSIS.md`, which locates it in the score offset instead.
- **Only linear structure was examined.** A nonlinear probe might recover scanner identity
  from the projected features just as easily, and an adversarially trained encoder might
  remove it where a projection cannot. Neither was tested.
- **Three scanners, one cohort, one country.** No general claim about acquisition hardware
  follows.

## 6. The useful conclusion

Put the two findings together and they rule something out cleanly:

> **The scanner disparity in this system is not a linearly removable representational
> shortcut.** The scanner signal cannot be projected away, and the TB head does not use the
> part of it that can be. Representation-level nuisance removal is therefore not the lever
> worth pulling here.

That is a negative result about a popular class of method, obtained cheaply, and it
redirects effort to where the disparity actually sits: the operating point.
