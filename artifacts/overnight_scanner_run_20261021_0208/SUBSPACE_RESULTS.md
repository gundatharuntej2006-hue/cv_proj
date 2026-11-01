# Task 3 — Preregistered scanner-subspace removal

## Outcome: **REJECTED** against its own preregistered criteria.

Protocol: `../overnight_run_20261021_2340/PRIMARY_EXPERIMENT_PREREGISTRATION.md` §4,
frozen in `FROZEN_MANIFEST.md` §2c before REPORT was scored.
Tables: `tables/subspace_tune.json`, `tables/subspace_metrics.csv`,
`tables/acceptance_verdict.json`, `tables/scanner_subspace_basis.npy`

---

## 1. The pipeline was proved equivalent before any new number was believed

Host RAM was 0.33–1.84 GB of 16.4 GB while the TBX11K regional cache alone is 1.35 GB, so
the committed `np.load` path was not usable. Features were streamed from the npz in
256-row blocks straight to the GPU instead.

That substitution is only legitimate if it computes the same thing, so it was checked
three ways:

| check | result |
|---|---|
| streamed tensor vs committed `repool(X, 12, 12)` on a 192-case subset | **100.000000% elementwise identical**, max absolute difference **0.0** |
| TBX11K development AUROC, 5-seed ensemble | **0.9368** against published 0.9368 |
| Vietnam TUNE AUROC, unprojected | **0.7237** against published 0.7237 |

Peak VRAM 1.48 GB against a 5.15 GB cap. Host RSS rose by 185 MB while loading a 555 MB
file. The script refuses to report any derived result if the baseline fails to reproduce.

## 2. What was fitted

A 3-class linear layer on Vietnam **TUNE** token vectors predicting `rad_system`, seed 17,
60 epochs, AdamW lr 1e-2. The class-mean-centred weight matrix was decomposed by SVD and
the leading **rank-3** right-singular subspace projected out of every token before pooling.
**No tuberculosis label entered the transform.** Orthonormality residual 4.8e-07.

### A defect in the preregistered rank, recorded before REPORT was read

The singular values came out **[73.94, 63.32, 0.00]**. A 3-class centred weight matrix has
rank 2, so the preregistered rank of 3 removes two genuine scanner directions plus **one
arbitrary direction from the null space**. The preregistration specified 3, so 3 is what
was scored and judged — correcting it after the fact is exactly what preregistration
exists to prevent. The practical effect is nil: rank 2 and rank 3 give TUNE AUROC 0.7234
and 0.7233 respectively.

The error came from the preregistration's own arithmetic, *"the number of scanner classes
minus one, plus one"*. For K classes the between-class subspace has rank K−1 = 2, not 3.

## 3. Results on REPORT — one sealed pass

n = 1,356, 141 TB. Paired bootstrap, 4,000 resamples, seed 0, identical cases.

| arm | group | AUROC | Δ vs baseline | 95% CI | |
|---|---|---|---|---|---|
| **P0 baseline** | pooled | 0.7242 | — | — | |
| | rad_1 | 0.7190 | | | |
| | rad_2 | 0.7890 | | | |
| | rad_3 | 0.6745 | | | |
| **P1 preregistered** | pooled | 0.7240 | **−0.0002** | [−0.0009, +0.0004] | includes 0 |
| | rad_1 | 0.7190 | +0.0000 | [−0.0019, +0.0017] | includes 0 |
| | rad_2 | 0.7891 | +0.0001 | [−0.0007, +0.0009] | includes 0 |
| | rad_3 | 0.6745 | +0.0001 | [−0.0014, +0.0017] | includes 0 |
| **P2 exploratory** | pooled | 0.7240 | −0.0002 | [−0.0009, +0.0005] | includes 0 |

P2 (head retrained in the projected space) is **not** covered by the preregistration and
cannot be declared a success whatever it shows. It shows nothing: the same null.

## 4. The acceptance criteria, applied as written

> *Accept only if the worst-scanner AUROC improves **and** the paired pooled difference
> excludes zero.*

| criterion | value | pass? |
|---|---|---|
| 1 — worst-scanner AUROC improves | 0.67447122 → **0.67452383** | technically **yes** |
| 2 — paired pooled CI excludes zero | −0.0002 [−0.0009, +0.0004] | **no** |
| **both required** | | **REJECT** |

**Criterion 1's "pass" is an artifact and is reported as such.** The worst-scanner AUROC
rose by **+0.000053** — about one thousandth of the ±0.055 per-scanner noise floor measured
on this cohort, and invisible at any sane reporting precision. Treating that as an
improvement would be absurd. The criterion was written without a minimum effect size; that
is a flaw in the preregistration, noted here rather than silently patched. Criterion 2
fails cleanly, so the overall verdict is REJECT either way.

## 5. What the null actually tells us

This is a null result, but not an empty one.

**Removing the linearly-decodable scanner directions changes tuberculosis discrimination by
essentially nothing — pooled, and on every individual scanner.** The largest movement
anywhere in the table is 0.0003.

Two readings are consistent with that, and the companion diagnostic
(`REPRESENTATION_DIAGNOSTICS.md`) separates them:

1. *The head never relied on those directions.* Supported — TB AUROC is flat even after
   iterative removal of 44 directions.
2. *The projection failed to remove the information.* Also partly true — scanner balanced
   accuracy falls only from 0.9896 to 0.9763 at the preregistered rank, and only to 0.9118
   after 44 directions. Scanner identity is far too redundantly encoded for a low-rank
   linear projection to erase.

So the honest conclusion is: **a low-rank linear nuisance projection is the wrong tool
here, and it was never load-bearing for the TB decision anyway.** The scanner disparity
documented in this project is not a linearly removable shortcut in the representation.
`CALIBRATION_AND_ERROR_ANALYSIS.md` shows where it actually lives.

## 6. Limitations

- One cohort, three scanners, one country.
- Vietnam TUNE chose the pooling configuration in earlier work, so REPORT is a controlled
  comparison, **not independent validation**.
- Only *linear* scanner structure was removed. A nonlinear or adversarial method could
  behave differently; this result does not speak to those.
- The transform was fitted on TUNE and applied to REPORT without refitting, which is
  correct, but it means the subspace was estimated from 1,352 cases.

## 7. Reproduction

```
py -3 artifacts/overnight_scanner_run_20261021_0208/scripts/verify_equivalence.py
py -3 artifacts/overnight_scanner_run_20261021_0208/scripts/task3_subspace_tune.py
py -3 artifacts/overnight_scanner_run_20261021_0208/scripts/task_report_sealed.py
```

Basis `tables/scanner_subspace_basis.npy`, sha256 `84c7a540805b4c91`. Seeds 17/29/43/61/79
for the head, 17 for the subspace. **Nothing here is deployable on the strength of one
exposed exploratory cohort, and the preregistered verdict is rejection.**
