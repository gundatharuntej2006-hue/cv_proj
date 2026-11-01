# Phase 4 — RayDINO benchmark: assessment and blockers

**Not run. Still blocked, for the same reason as three days ago.** A complete,
tested Kaggle-ready package already exists; it needs a licence decision and two facts,
not more code.

---

## 1. The package that already exists

`artifacts/kaggle_concept_raydino_20261021_1830/` — built 2026-10-21, never executed.

| | |
|---|---|
| integrity | `MANIFEST.sha256` re-verified today: **28 of 29 files match byte-for-byte**. The one mismatch is `FINAL_RUN_REPORT.md`, which was written *after* the manifest and describes it — expected, not corruption |
| tests | **29 of 29 pass** today (`py -3 -m pytest artifacts/kaggle_concept_raydino_20261021_1830/tests -q`, 23 s) |
| protocol | `RAYDINO_COMPARISON_PROTOCOL.md`, hashed before any encoder was loaded |
| notebooks | `kaggle/00_preflight.ipynb`, `02_raydino_vs_raddino.ipynb` |
| guards | `src/guards.py` — `assert_target_mapping`, `assert_ids_aligned`; `evaluate.fit_threshold` raises for any partition other than development |

Writing a second package would duplicate work that is already done and tested. This phase
verifies it instead.

## 2. Facts, each marked verified / recorded / unknown

| fact | status |
|---|---|
| RAD-DINO revision `110cbc18d5133582e320b43d53bf5c44e410c936`, MIT | **verified today** — loaded and used in this run |
| RAD-DINO exposure to TBX11K / Vietnam / Shenzhen / Montgomery / DA / DB | **UNKNOWN.** The model card does not name its checkpoint-selection datasets. Recorded as UNKNOWN in the project exposure registry and not guessed |
| RayDINO identity: `xray_dino_vitl16` via `facebookresearch/dinov2` | **recorded 2026-10-21**, not re-verified today |
| RayDINO licence: **FAIR Noncommercial Research License, gated weights** | **recorded 2026-10-21**, not re-verified today. No attempt was made to fetch gated weights |
| RayDINO architecture: ViT-L/16, 1024-d | **recorded as expected, asserted at runtime** — not confirmed from a model card |
| RayDINO input resolution and patch grid | **UNKNOWN.** Not published in a form the earlier audit could pin. The protocol declares it must be read from the model card at run time, never assumed |
| RayDINO exposure to any of this project's evaluation cohorts | **UNKNOWN** |

Nothing above was re-fetched today. A three-day-old licence reading is stated as a
three-day-old licence reading.

## 3. Blockers, in the order they must be cleared

1. **Licence — the hard one.** The FAIR Noncommercial Research License is a
   non-commercial, gated licence. Accepting it is an institutional decision about a
   student project that may be published or demonstrated, not a technical one, and it is
   **not mine to take**. Downloading gated weights requires accepting terms on someone's
   behalf.
2. **Redistribution terms are unresolved.** Running on Kaggle means putting the weights
   into a Kaggle dataset. Whether the licence permits that, even privately, has not been
   established. **Do not upload the weights to Kaggle until it has been.**
3. **Input resolution is unpublished.** 16 does not divide 518, so the patch grids cannot
   match RAD-DINO by construction. The protocol already handles this by making matched
   **global** heads the primary comparison and the 12x12 regional pooling an explicitly
   declared secondary — but the actual resolution still has to be read from the card
   before extraction.
4. **Checkpoint exposure is unknown for both encoders.** A difference between them could
   be a difference in pretraining exposure to these very cohorts. This limits what any
   result could mean and must appear in the write-up whatever the outcome.

## 4. Target label for the benchmark — settled by Phase 2

The brief asks that RayDINO use the target chosen in Phase 1. Phase 1 established that
`active_tb_label` is the **documented** active-TB target; Phase 2 then tested it and
**rejected** it for internal development (primary endpoint -0.0037, interval excluding the
declared minimum effect).

**So the benchmark target stays `tb_label`** — which is exactly what
`configs/target_label.yaml` already pins, with the mismatch declared in the config's own
`description` field rather than silently inherited. No change is needed, and changing it
on the strength of an exploratory Vietnam-only gain would be the retrospective promotion
this project keeps refusing to do.

## 5. Priority

**Lower than everything in Phase 3 §4.** A new encoder is a bet that the representation
is the bottleneck. The evidence does not point there: the same frozen RAD-DINO features
give AUROC 0.95 on TBX11K calibration and 0.72 on Vietnam. A gap of that size between an
internal and an external cohort, with the same features and the same head, is a
distribution and task-definition problem far more than a representation problem — and
Phase 2 just showed that a target-definition fix worth +0.02 externally is worth nothing
internally.

Swapping encoders would not test that. It is worth doing eventually, as a controlled
negative control on the representation hypothesis, after the concept axes are understood
and the definitive C01 split exists.

## 6. Manual steps, if the licence is cleared

1. Accept the FAIR Noncommercial Research License and confirm in writing whether it
   permits a **private** Kaggle dataset copy. If it does not, stop — run locally or not at
   all.
2. Read the model card for the **exact** revision, input resolution and patch size; record
   them in `manifest_template.json`. Do not proceed on an assumed resolution.
3. Upload the weights to a **private** Kaggle dataset. No images beyond what the existing
   `KAGGLE_INPUT_CHECKLIST.md` already lists. **No Vietnam REPORT, no `internal_test`.**
4. Run `kaggle/00_preflight.ipynb` and stop if any assertion fails.
5. Run `kaggle/02_raydino_vs_raddino.ipynb`. Both arms re-extract their own features —
   cached RAD-DINO tokens must **not** be reused for RayDINO.
6. Download `tables/encoder_comparison.csv` and the manifest; transcribe into a new
   timestamped artifact directory. Apply the decision rule in
   `RAYDINO_COMPARISON_PROTOCOL.md` §6 unchanged.

Until step 1 has an answer, this phase stays where it is.
