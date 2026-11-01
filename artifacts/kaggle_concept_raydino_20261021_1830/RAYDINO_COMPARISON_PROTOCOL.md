# Stage 2 — RayDINO vs RAD-DINO: protocol

**Written and hashed BEFORE any encoder was loaded or any feature computed.** Hash in
`MANIFEST.sha256`. **No fit has been performed.** This is the pre-registered design.

**Direction is not assumed. RayDINO may improve, match or worsen performance.**

---

## 1. Question

Under a matched downstream head, identical training rows and an identical target label,
does a different frozen chest-radiograph representation change TB discrimination?

## 2. Arms

| | arm A | arm B |
|---|---|---|
| encoder | **RAD-DINO**, `microsoft/rad-dino` rev `110cbc18d5133582e320b43d53bf5c44e410c936` | **RayDINO / XRay-DINO**, `xray_dino_vitl16` via `facebookresearch/dinov2` |
| arch / dim | ViT-B/14 / 768 | ViT-L/16 / 1024 expected — **asserted at runtime** |
| input / grid | 518 px / 37×37 | **unknown — read from the model card** |
| weights licence | MIT | **FAIR Noncommercial Research License, gated** |
| state | **frozen** | **frozen** |

Both arms share: the same eligible images, the same target mapping, the same train and
development rows, the same head class, the same regularisation grid, the same seeds and
the same evaluation code. **The encoder is the only deliberate difference.**

## 3. The comparability problem, declared not hidden

16 does not divide 518, so **the patch grids cannot match by construction**, and the
embedding dimensions differ (768 against 1024). Therefore:

- **PRIMARY: matched global heads.** Each encoder's own pooled output, under its own
  official model-card preprocessing.
- **SECONDARY: regional.** Via `encoders.pool_to_grid` — adaptive average pooling to a
  common 12×12 grid then L2-normalisation — declared here in advance and applied
  identically to both arms. If it cannot be implemented cleanly for RayDINO, the regional
  comparison is reported **unavailable**, not approximated.
- **Preprocessing differs between arms and is reported as a factor and a limitation.** It
  is never presented as identical, and it is never tuned — least of all against REPORT or
  the locked partition.

Cached RAD-DINO tokens **cannot** be reused for RayDINO. Images are re-encoded for each.

## 4. Data and partitions

Head fitted on **TBX11K train** (5,941 rows, filtered to the target categories).
Regularisation chosen on **TBX11K development** (659). Threshold at 70% specificity on
**development** only — `evaluate.fit_threshold` raises for any other partition.

**`internal_test` (901) is never touched. Vietnam REPORT is never read.** Vietnam TUNE may
be scored for diagnostic comparison and is labelled **exploratory and exposed**.
Shenzhen, Montgomery and DA/DB are **not used**.

## 5. Target label

`tb_label == 1`, identical in both arms. `latent_tb` is positive here despite its own
`active_tb_label = 0` — a documented limitation of the reference pipeline, deliberately
not corrected in one arm only. `configs/target_label.yaml`;
`guards.assert_target_mapping` refuses an undefined mapping.

## 6. Primary endpoint and decision rule

**Primary:** AUROC on TBX11K development, matched global heads, RayDINO minus RAD-DINO,
paired bootstrap (4,000 replicates, seed 0) on identical cases.

**Minimum meaningful effect: +0.02 AUROC**, fixed now.

**ADOPT-FOR-FURTHER-STUDY requires all three:** Δ ≥ +0.02; the 95% paired CI excludes
zero; and **no unacceptable safety loss** — sensitivity at 70% specificity must not fall,
particularly in the **previous-TB subgroup**, where discrimination is already weakest
(TUNE: 0.6957 separation among negatives; see `CONCEPT_AUDIT_REPORT.md`).

Anything else is **REJECT** or **INCONCLUSIVE**. A favourable point estimate with an
interval containing zero is not a positive result. A gain on Vietnam TUNE alone is not a
win.

**No fine-tuning.** LoRA or full fine-tuning is a separate later experiment, only if the
frozen comparison gives a reason.

## 7. Reported outputs

`tables/encoder_comparison.csv`: per split and per arm — n, positives, prevalence, AUROC
with CI, AUPRC with its prevalence baseline, sensitivity at 70% specificity, specificity
at 90% sensitivity, previous-TB subgroup metrics with sample sizes, extraction and head-fit
time, peak GPU memory, feature storage size. Plus the paired ΔAUROC with its CI, and the
fitted head weights stored as `.joblib`.

## 8. Reproducibility and integrity

Seeds pinned (head 17, bootstrap 0). Versions pinned. Checkpoint revisions in
`manifest.json`. Features carry a row-ID manifest, shape assertions and a SHA-256. **Row
alignment is proved by stable case IDs via `guards.assert_ids_aligned`, never by
positional order.** Extraction is resumable and halves its batch on OOM without changing
semantics.

## 9. Status

**NOT RUN.** Blocked on gated RayDINO weights, their unresolved redistribution terms, and
the unpublished input resolution. Nothing is reported as a result.
