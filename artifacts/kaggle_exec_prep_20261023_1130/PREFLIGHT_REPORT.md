# Preflight report

Verified 2026-10-23 on this machine. Machine-readable copy:
`tables/preflight_local.json`. **Verdict locally: STOP**, on one blocker — host RAM.
Everything else is clear.

---

## 1. Verified prerequisites

| prerequisite | status | evidence |
|---|---|---|
| **Data rights** | **CC0-1.0** | the dataset-specific Dryad API field for `doi:10.5061/dryad.5tb2rbpcv`. See `DATA_RIGHTS_ASSESSMENT.md` in the previous package. CC0 waives copyright only |
| **CLEAR licence** | **Apache-2.0** | repository `LICENSE` and HF card metadata agree |
| **Pinned revision resolves** | **yes** | `HfApi.repo_info(revision=142f169c…)` returns `sha = 142f169c8b006b0f28cf55c6a237c3003db5ed8f`. Metadata only; nothing was downloaded |
| **Asset identity** | **pinned and verified** | checksums below, now in `configs/model_assets.json` and compared by preflight |
| **Cohort** | **1,352 TUNE cases**, 139 TB-positive, 466 previous TB | fingerprint `1aa55cd5c0eb2088…` matches `configs/tune_cohort_fingerprint.json` |
| **Images** | **1,352 approved, 0 missing** | every allow-listed id has a file |
| **The real zipped upload** | **resolves, 1,352 members, 0 unapproved** | preflight run against `D:\tb-triage-kaggle-staging\concept-audit-tune.zip` (173 MB) on 2026-10-24: layout `staged_zip`, fingerprint matched, no forbidden member. Images are read from the archive; nothing is extracted |
| **Concept mapping** | **9 of 12 exact**, 3 excluded | `configs/concept_indices.json` |
| **Expected output shape** | **(1352, 9)** | asserted before analysis |
| **Dependencies** | all present, all matching the pins | numpy 2.2.6, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.17.1, torch 2.6.0+cu124, torchvision 0.21.0+cu124, Pillow 12.3.0, huggingface_hub 1.21.0, psutil 7.2.2 on Python 3.13.2 |
| **GPU** | present locally | RTX 4050 Laptop, 5.32 GB VRAM free. Not the constraint |
| **Disk** | 184.7 GB free | against a 6 GB minimum |
| **Partition boundary** | **clean** | no module names the locked partition; no forbidden path among the inputs |
| **Tests** | **49 passed** (new package) and **20 passed** (previous package, re-run unchanged) | `py -3 -m pytest .../tests -q` |

### Asset checksums, as pinned

| file | bytes | sha256 |
|---|---|---|
| `best_model.pt` | 603,060,057 | `da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03` |
| `concept_embeddings_368294.pt` | 1,131,400,935 | `7241c5b920c4394b507f2c4bd89f1d0b70775399caaa2f5323cb373afb1a00d6` |
| `mimic_concepts.csv` | 22,208,546 | `93e975b722422b64f18b57161ac9894fe30ac7a04cc68b7aa8a2e01f6d0fc439` |

That last line matters beyond provenance: it is the same hash as the vocabulary file the
concept indices were mapped against, so **the 9 pinned indices refer to the vocabulary
that will actually be loaded.** Preflight now compares the live values against these and
raises a blocker on a mismatch rather than reporting it as a note.

## 2. Remaining blocker — host RAM, locally only

```
host RAM available is 0.71 GB, below the 3.0 GB needed to hold the CLEAR
checkpoint and a slice of the concept tensor. Run on Kaggle, or free memory.
Do NOT reduce the image resolution or swap the architecture to fit.
```

Measured repeatedly across the session: **0.58–1.48 GB available of 16.39, 94–96 % used.**
The requirement is not a guess — the checkpoint is 603 MB and the concept tensor 1,131 MB,
and torch needs room to unpack them. **Kaggle clears this blocker**; so would freeing
memory locally. The resolution and the architecture are not negotiable, and preflight will
not let them be traded away to fit.

## 3. Internet is required. A fully offline run is not possible.

Two fetches happen at run time:

1. the pinned CLEAR assets, from `huggingface.co`;
2. the DINOv2 ViT-B/14-with-registers **architecture**, via `torch.hub` from `github.com`.

Both are stated in `preflight.run()["network_required"]` and asserted by test. Pre-attaching
both as Kaggle datasets and repointing the loaders would make an offline run possible, but
**this package is not written that way**, so claiming otherwise would be false. Turn
Internet on.

## 4. The checkpoint key layout — RESOLVED, and it found two further errors

**Status: resolved on 2026-10-24, on Kaggle, by the fail-closed check doing its job.**

The run reached inference and stopped with:

```
best key mapping None matched 0.0% of the encoder's tensors
WeightLoadError: only 0.0% of the DINOv2 ViT-B/14-reg parameters could be matched
  checkpoint top-level keys: ['positional_embedding', 'text_projection', 'logit_scale',
    'visual.backbone.cls_token', 'visual.backbone.pos_embed', ...]
```

The prefix is **`visual.backbone.`**, which the guesswork did not try. Reading CLEAR's
own `src/clear/hub.py` and `src/clear/pipeline.py` then showed the hand-rolled encoder was
wrong in **three** ways, not one:

| | what was wrong | would it have raised? |
|---|---|---|
| key prefix | `visual.backbone.` not among the prefixes tried | **yes** — and it did |
| **projection** | the visual tower is backbone **+ `nn.Linear` projection**, and `encode_image` returns the **projected** vector. The raw backbone output is in a different space from the concept embeddings | **no** |
| **preprocessing** | CLEAR uses a **bicubic** resize to 448 and **CXR** statistics (mean 101.48761/255 ≈ 0.398, std 83.43944/255 ≈ 0.327); ImageNet constants and a bilinear resize were used | **no** |

Only the first would have stopped anything. The other two would have produced a complete,
plausible-looking table of meaningless numbers — which is the failure this whole package
exists to prevent, and which the 90% threshold would not have caught.

**The fix is to stop reimplementing the loader.** `_load_encoder` now calls
`clear.load_pretrained(checkpoint_path=...)`, pinned to commit
`57d2eae57ddabf8e56c8013ec675336d71b3d8e7`, installed with `--no-deps` so CLEAR's
dependency pins do not reinstall torch. Preprocessing is CLEAR's own
`build_cxr_preprocess`, returned by the same call, so it cannot drift. The run still
refuses if the loaded model lacks `encode_image` or a visual `projection`.

A test asserts the encoder is not rebuilt: no ImageNet constants, no `torchvision.transforms`,
no direct `torch.hub` DINOv2 call, and `encode_image` is what gets used.

## 5. Preprocessing fidelity — RESOLVED

Previously recorded here as an unconfirmed caveat. It was not merely a caveat: it was
wrong, and it is now CLEAR's own transform rather than a reconstruction.

## 6. What is not in this run

Not attempted, by instruction and by design: Vietnam REPORT, TBX11K `internal_test`,
RayDINO (gated, FAIR Noncommercial Research Licence), feature subtraction, apical
weighting, subgroup Platt scaling, retraining, threshold optimisation. No production code,
checkpoint, prediction file or earlier report was modified.

## 7. Verdict

**Confirmed GO on Kaggle**: preflight passed there on 2026-10-24 with all three asset
checksums matching the pinned values, 1,352 cases, fingerprint matched, 0 missing images.
STOP locally, on RAM alone. The one risk that survived into the run (§4) fired exactly as
intended and has been resolved.
