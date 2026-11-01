# Model licence assessment

Verified 2026-10-23 from official sources. **Data licensing and model licensing are
separate questions.** The Dryad CC0 finding resolves the first and says nothing about
the second.

---

## CLEAR — cleared, and pinned

| | |
|---|---|
| code | `github.com/peterhan91/CLEAR`, **Apache-2.0** (`LICENSE`), default branch `main` |
| weights and assets | `huggingface.co/peterhan91/CLEAR`, card metadata **`license: apache-2.0`** |
| **pinned revision** | **`142f169c8b006b0f28cf55c6a237c3003db5ed8f`** (HF repo sha, lastModified 2026-06-12) |
| architecture | DINOv2 ViT-B/14 **with registers** + a 12-layer GPT-2-style text encoder, shared 768-d space |
| input | **448 x 448** |
| assets | `best_model.pt` (603 MB) · `concept_embeddings_368294.pt` (1,131 MB) · `mimic_concepts.csv` (22.2 MB) · `concepts_embeddings_sfr_mistral.pickle` |
| output | per-concept **similarities** over 368,294 observations, and a 768-d image embedding |
| dependencies | Python >= 3.10, PyTorch, `torch.hub` for the DINOv2 architecture |
| GPU requirement | **not stated by the authors.** Measured need here is host RAM, not VRAM |

**A caveat the repository states itself:** the README notes that model weights, concept
assets, and upstream models or datasets "may have separate terms" from the Apache-2.0
code. The HF card metadata gives `apache-2.0` for the weights repository, so code and
weights agree. The upstream training corpora (MIMIC-CXR, CheXpert-Plus, ReXGradient) carry
their own terms, which govern those datasets and not our inference.

**No commit hash is displayed on the GitHub page.** The HF repo sha above is what has been
pinned; if the GitHub inference code is also used, pin its commit separately before
relying on it.

## RayDINO — untouched, as required

**Nothing was downloaded, no form was bypassed, and no weight was re-hosted.** Weights
remain under the **FAIR Noncommercial Research License**, gated behind a request form at
`ai.meta.com/resources/models-and-libraries/raydino-downloads/`; the DINOv2 loader code is
Apache-2.0. **RayDINO benchmarking is out of scope for this task** and remains blocked on
its licence and access.

## RAD-DINO — unchanged

`microsoft/rad-dino`, revision `110cbc18d5133582e320b43d53bf5c44e410c936`, **MIT**. Used
only as the already-frozen source of the committed classifier scores. **Not retrained, not
re-run, not modified.**

## Exposure — still unknown for all three

None of RAD-DINO, CLEAR or RayDINO can be said not to have seen TBX11K, Shenzhen,
Montgomery, DA/DB or this Vietnam cohort. CLEAR's card names three training corpora and
no TB benchmark, but names no evaluation set either, so exposure is **unknown rather than
excluded**. If CLEAR's published evaluation does overlap this cohort, any concept
association measured here is **exploratory and potentially exposed**, and CLEAR's
published performance is not independent validation of anything in this project.

## The distinction that matters

| question | answer | source |
|---|---|---|
| May we analyse the Vietnam images? | **Yes** — CC0-1.0 | Dryad API, dataset-specific |
| May we redistribute them? | **Yes** legally; **private dataset recommended** | Dryad reuse guidance |
| May we run CLEAR on them? | **Yes** — Apache-2.0, pinned | HF card + repo LICENSE |
| May we run RayDINO on them? | **Not yet** — gated weights, NC licence, redistribution unresolved | Meta form + `LICENSE_XRAY_DINO_MODEL` |
| Does CC0 on the data imply anything about the models? | **No.** Entirely separate instruments | — |
