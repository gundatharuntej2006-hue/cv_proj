# Source list and verification status

**Owner:** Person 2 · **Date:** 2026-10-21

Every source below is marked by whether it was **actually retrieved and read in this
environment**. Sources not retrieved are not summarised as if read, and no claim in any
project document rests on an unverified source.

| # | source | URL | status | what was taken from it |
|---|---|---|---|---|
| 1 | **RADAR** — *Acquisition-Adversarial Attention Pooling over a Frozen Chest-Radiograph Foundation Model for Tuberculosis Screening*, MICCAI 2026 satellite proceedings | https://papers.miccai.org/miccai-2026-sat/TREAT-MMTB_012.html | **RETRIEVED AND READ** | Method: frozen CXR foundation model, multi-depth patch representations, learned TB query cross-attending over patches, modality- and conditional-adversarial objectives, style randomisation with prediction consistency, seed/view ensembling, higher-resolution read-out for small focal abnormalities, no acquisition metadata at inference. Results on the page: internal F1 0.9840, external F1 0.8375, worst-site F1 0.8815 → 0.9064 across five held-out public sites (not named on the page). |
| 2 | **CheXFound / GLoRI** — *Chest X-Ray Foundation Model With Global and Local Representations Integration*, IEEE TMI 44(12), 2025 | https://doi.org/10.1109/TMI.2025.3581907 · full text https://pmc.ncbi.nlm.nih.gov/articles/PMC12790848/ | **NOT RETRIEVED** in this environment | Cited only as context for the claim that global+local feature integration is an established CXR foundation-model adaptation strategy. Its primary task setting differs from this TB experiment. **No specific result or method detail is attributed to it.** |
| 3 | **RAD-DINO model card** | https://huggingface.co/microsoft/rad-dino | **RETRIEVED AND READ** | Training data: MIMIC-CXR (368,960), CheXpert (223,648), NIH-CXR (112,120), PadChest (136,787), BRAX (41,260) = 882,775. Shenzhen/Montgomery not listed in training. **The released checkpoint was selected by linear probing on validation sets of "the evaluation datasets described in the paper", which the card does not name** — so checkpoint-level exposure to common public TB benchmarks cannot be excluded. |
| 4 | **Domain-Aware TB screening with a LoRA-adapted CXR foundation model**, MICCAI 2026 satellite | https://papers.miccai.org/miccai-2026-sat/TREAT-MMTB_011.html | **PAGE RETRIEVED**, but **title, authors and venue not independently verified** beyond the page itself | Uses frozen RAD-DINO and CheXFound with LoRA; reports Shenzhen F1 0.7707 and Montgomery F1 0.8348. **Per the task constraint, this is not cited in `paper_draft.md`**, because exact title/authors/venue could not be confirmed from the retrieved source. It is recorded here only. |
| 5 | **Shortcut learning in medical AI hinders generalization**, npj Digital Medicine, 2024 | https://www.nature.com/articles/s41746-024-01118-4 | **NOT RETRIEVED** | Not relied upon for any specific claim. The project's shortcut findings rest on its own measurements. |

## Rules applied

- A source marked NOT RETRIEVED is never paraphrased as if read, and never used to support a
  specific numerical or methodological claim.
- Source 4 is deliberately excluded from the paper draft under the constraint that its
  bibliographic details could not be verified from the retrieved page.
- Only sources 1 and 3 are cited in `report/paper_draft.md`. Source 2 appears there
  explicitly flagged as not retrieved.

## Correction recorded

In conversation I earlier asserted that the field generally discards patch tokens and that
exploiting them is "a free fix any ViT-based CXR system could adopt". **That claim is wrong**
— RADAR and GLoRI both exploit patch-level representations — and it is contradicted by
source 1, which was subsequently retrieved.

**It was never written into any repository file.** A search of `docs/` and `report/` for
that wording returns nothing; it existed only in chat. No file correction was required for
it, and none is claimed. It is recorded here so the error is not silently dropped.
