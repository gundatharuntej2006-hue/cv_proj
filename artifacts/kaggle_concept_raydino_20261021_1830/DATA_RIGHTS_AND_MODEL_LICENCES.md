# Data rights and model licences

**Unknown means BLOCKED, never "presumed allowed".** Verified 2026-10-21 from official
sources. Code and model-weight terms are recorded separately where they differ.

---

## Datasets

| component | official source | stated terms | Kaggle use permitted? | unresolved |
|---|---|---|---|---|
| **Vietnam CXR images** (2,708) | Dryad [10.5061/dryad.5tb2rbpcv](https://doi.org/10.5061/dryad.5tb2rbpcv) | The record states only *"Content on this site is licensed for reuse"* and **names no specific licence**. Not CC0, not CC-BY — **no licence identifier is given** | **NO — BLOCKED** | Whether redistribution to a third-party platform is permitted. Ask the depositors (Codlin et al.) or Dryad support before any upload |
| **Vietnam participant table** | same | same | **NO — BLOCKED** | patient-level table; same question, higher sensitivity |
| **TBX11K images** | TBX11K archive | not verified in this run | **NO — unverified, therefore blocked** | redistribution terms |
| **Derived embeddings** (RAD-DINO features) | this repository | derived from the above images | **NO — BLOCKED** | a derived representation inherits the source restriction until shown otherwise |

**Consequence:** the Kaggle bundle is **code-only**. The user attaches any data privately
and themselves, having satisfied themselves it is permitted. Nothing in this package
uploads data, and no Kaggle API credential is used or read.

## Model weights and code

| component | official source | code licence | weights licence | Kaggle use | unresolved |
|---|---|---|---|---|---|
| **RAD-DINO** | [huggingface.co/microsoft/rad-dino](https://huggingface.co/microsoft/rad-dino), revision `110cbc18d5133582e320b43d53bf5c44e410c936` | MIT | **MIT** | **yes** — downloadable with internet enabled | none |
| **CLEAR** | [huggingface.co/peterhan91/CLEAR](https://huggingface.co/peterhan91/CLEAR); code [github.com/peterhan91/CLEAR](https://github.com/peterhan91/CLEAR) | Apache-2.0 | **Apache-2.0** | **yes** — downloadable with internet enabled | no commit hash is published on the model card; **pin one before use** |
| **RayDINO / XRay-DINO** | request form [ai.meta.com/.../raydino-downloads](https://ai.meta.com/resources/models-and-libraries/raydino-downloads/); loader in [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Apache-2.0 (DINOv2 code) | **FAIR Noncommercial Research License** (`LICENSE_XRAY_DINO_MODEL`) | **UNRESOLVED** | **Weights are GATED** — access by form, temporary emailed link. Whether re-hosting them as a Kaggle dataset counts as permitted redistribution under the FAIR NC licence is **not established**. Read the licence before attaching. Non-commercial research use of this project appears to fit the intended use, but redistribution is a separate question |

## Architecture and preprocessing, as published

| | RAD-DINO | CLEAR | RayDINO |
|---|---|---|---|
| backbone | ViT-B/14 | DINOv2 ViT-B/14 + registers | ViT-L/16, 307M params |
| embedding dim | 768 | 768 (shared image-text space) | 1024 expected — **assert at runtime** |
| input resolution | 518 px | **448 px** | **UNKNOWN — read from the model card; 16 does not divide 518** |
| training data | MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX (882,775 images) | MIMIC-CXR, CheXpert-Plus, ReXGradient (~0.87M image–report pairs, 239,391 patients) | 873k chest X-rays |

## Cohort exposure of each encoder

| encoder | TBX11K | Shenzhen | Montgomery | DA/DB | Vietnam |
|---|---|---|---|---|---|
| **RAD-DINO** | **unknown** | **unknown** | **unknown** | **unknown** | **unknown** |
| **CLEAR** | not named | not named | not named | not named | not named — but **unknown**, not excluded |
| **RayDINO** | **unknown** | **unknown** | **unknown** | **unknown** | **unknown** |

**No cohort may be called "unseen" by any of these encoders.** RAD-DINO's model card
states its released checkpoint was selected by linear probing on validation sets of
"the evaluation datasets described in the paper", which it does not name — so
checkpoint-selection exposure to public TB benchmarks cannot be ruled out. CLEAR's card
names three training corpora and no TB benchmark, but names no evaluation set either, so
exposure is **unknown rather than excluded**. RayDINO's training corpus is described only
by size. If CLEAR's published evaluation turns out to include this Vietnam library or
overlapping patients, its concept associations here are **exploratory and potentially
exposed**, and its published performance is not independent validation of anything here.

## Decisions taken

1. **Code-only ZIP.** No image, patient table, embedding or pretrained weight is included.
2. **The user attaches data manually**, having checked permissions themselves.
3. **The concept audit is gated in code** on an explicit manifest flag; unknown blocks it.
4. **RayDINO weights are never fetched by this package** — the loader raises unless the
   user supplies a path or the gated URL.
5. **Pin a CLEAR commit** before relying on its concept bank; none is published.
