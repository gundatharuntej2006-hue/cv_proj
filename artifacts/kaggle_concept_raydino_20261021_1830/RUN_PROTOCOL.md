# Run protocol

Ordered, with the gate that must pass before each stage. Hashed before any encoder loads.

---

## Order and gates

| stage | gate that must pass first | may run locally? | may run on Kaggle? |
|---|---|---|---|
| 0 preflight | none | **yes, done** | yes, notebook 00 |
| 1 concept audit, score-only | Stage 0 clean | **yes, done** (TUNE only) | yes |
| 1 concept audit, concept inference | images attached **AND** manifest clears the image licence | **no** — licence BLOCKED | only once unblocked |
| 2 encoder benchmark | target mapping validated, splits loaded with locked rows dropped, both encoders loadable | **no** — RayDINO weights gated | yes, once weights are supplied |
| 3 intervention | Stage 1 signal conditions **all four** met **AND** Stage 2 valid | no | **disabled by default** |

## Partition discipline

| partition | permitted use in this protocol |
|---|---|
| TBX11K train (5,941) | head fitting, both arms, identical rows |
| TBX11K development (659) | regularisation choice, thresholds, calibration — **and nothing else** |
| TBX11K calibration (899) | available, unused by default |
| **TBX11K internal_test (901)** | **LOCKED.** Row count from the manifest only |
| Vietnam TUNE (1,352) | exploratory diagnostics; **already exposed**, not independent validation |
| **Vietnam REPORT (1,356)** | **not read** |
| Shenzhen, Montgomery, DA/DB | **not used** in this run. If ever scored, descriptive only, labelled previously exposed, and never for selection |

## Target label, identical in both arms

`tb_label == 1`: positives are `active_tb`, `latent_tb`, `active_latent_tb`; negatives are
`healthy` only. `latent_tb` carries `tb_label=1` while its own `active_tb_label=0` — a
**documented limitation of the reference pipeline**, applied identically to both encoders
so the comparison stays fair. It is deliberately **not** corrected in one arm.
`guards.assert_target_mapping` refuses to train if the mapping is missing or ambiguous.

## Encoder comparability, declared not glossed

| | RAD-DINO | RayDINO |
|---|---|---|
| arch / dim | ViT-B/14 / 768 | ViT-L/16 / 1024 expected |
| input / patch grid | 518 px / 37×37 | **unknown — read from the model card** |
| preprocessing | official HF processor | official model-card transform |

Patch grids and dimensions **cannot** match. Therefore:

- **Primary: matched global heads**, each encoder under its own official preprocessing.
- **Secondary: regional**, via `pool_to_grid` — adaptive average pool to a common 12×12
  grid then L2-normalise, declared in advance and applied identically to both.
- **Preprocessing differences are reported as a factor and a limitation**, never presented
  as identical.
- Cached RAD-DINO tokens **cannot** be reused for RayDINO. Images are re-encoded.

## Reproducibility

Seeds pinned (head 17, bootstrap 0). Package versions pinned in `requirements.txt`.
Checkpoint revisions recorded in `manifest.json`. **Fitted head weights are stored**, not
just seeds. Features carry a row-ID manifest, shape assertions and a SHA-256. Row
alignment is proved by `guards.assert_ids_aligned` on stable case IDs — **never by
positional order**.

## Resumability

Feature extraction checkpoints every block beside its output `.npz`, so an interrupted
Kaggle session resumes without recomputation. OOM halves the batch size without changing
semantics. Elapsed time and peak GPU memory are logged.

## Stop conditions

- Any Stage 0 failure → stop, do not proceed.
- Image licence unresolved → concept inference does not run; the blocker is reported.
- RayDINO weights absent → that arm is **skipped, not faked**.
- RayDINO input resolution unknown → the loader raises rather than guessing.
- A prerequisite failure is recorded in the report; the run is **not** called complete.
