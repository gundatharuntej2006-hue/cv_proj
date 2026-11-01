# Prior art and novelty audit

**Owner:** Person 2 · **Date:** 2026-10-21 · **Repo HEAD:** `a78e73a` ·
**Type:** read-only literature and artifact audit. No code changed, nothing retrained.

## Headline verdict

**The architecture is not novel.** Frozen chest-radiograph foundation model + patch-level
focal aggregation for tuberculosis screening is directly covered by RADAR (MICCAI 2026
satellite), which additionally does the acquisition-invariance work we only diagnosed.
Global + local feature integration from a frozen CXR foundation model is covered by
GLoRI/CheXFound (IEEE TMI 2025). We should not claim architectural novelty, and the report
and paper draft must be corrected where they imply it.

What may still be defensible is the **evaluation protocol and the negative results**, not
the model. See §4.

## Sources actually fetched and read

| source | fetched | what was confirmed |
|---|---|---|
| RADAR, MICCAI 2026 sat, `TREAT-MMTB_012` | yes | frozen CXR foundation model, multi-depth patch features, learned TB query cross-attending over patches, modality- and conditional-adversarial heads, style randomisation, prediction consistency, seed/view ensembling, **higher-resolution read-out targeting small focal abnormalities**. Internal F1 0.9840, external F1 0.8375; worst-site F1 0.8815 → 0.9064. Five held-out public sites, not named on the page. |
| Domain-Aware TB screening, MICCAI 2026 sat, `TREAT-MMTB_011` | yes | frozen **RAD-DINO** and CheXFound with **LoRA**; reports **Shenzhen F1 0.7707, Montgomery F1 0.8348**. No AUC given on the page. |
| RAD-DINO model card | yes | trained on MIMIC-CXR, CheXpert, NIH-CXR, PadChest, BRAX (882,775 images). Shenzhen/Montgomery **not** in training. **Released checkpoint selected by linear probing on validation sets of evaluation datasets the card does not name.** |
| GLoRI / CheXFound, IEEE TMI 2025 | **not fetched** (DOI link not retrieved in this environment) | characterised from the task brief only: integrates global and fine/coarse-grained local features from a frozen CXR foundation model, evaluates on Shenzhen/Montgomery among others. **Treated as unverified.** |
| Shortcut learning, npj Digit Med 2024 | **not fetched** | not relied on for any specific claim here. |

**Limitation:** two of five sources were not retrieved. Conclusions that depend on GLoRI's
exact method are marked *unknown pending review* rather than asserted.

## Novelty matrix

| our component / claim | closest prior art | exact overlap | important differences | evidence | verdict |
|---|---|---|---|---|---|
| Frozen RAD-DINO for TB classification | Domain-Aware TB (MICCAI 2026 sat) | Same backbone, same task, frozen | They add LoRA adaptation; we keep it strictly frozen | `TREAT-MMTB_011` fetched | **already established** |
| Local/patch/regional features instead of global | RADAR; GLoRI | RADAR extracts multi-depth patch representations; GLoRI integrates local + global | We use single-depth 12×12 regions only | `TREAT-MMTB_012` fetched; GLoRI unverified | **already established** |
| Regional **max**-pooling specifically | RADAR (learned TB query cross-attention) | Both aggregate focal patch evidence | They learn attention; we take the max, and we **measured attention to be worse** (0.6416 vs 0.7239 on our tune half) | `attention_mil.json`; RADAR page | **incremental** — the finding that the simpler aggregator wins in a low-data regime is a small, specific contribution |
| Generalisation across institutions | RADAR (5 held-out sites); Domain-Aware (Shenzhen/Montgomery) | Cross-dataset TB evaluation is standard | We add a **consecutive screening cohort with Xpert reference and a commercial comparator**, which neither page reports | `p2_vietnam_result.md` | **potentially differentiating** (the cohort and comparator, not the method) |
| Explicit acquisition-shortcut **mitigation** | RADAR (adversarial + style randomisation) | — | **We do not mitigate.** We diagnose and quantify only | RADAR page | **already established, and we are behind it** |
| Shortcut **quantification** (border-only AUC 0.9094, collapses to 0.52–0.62 across 4 sites) | npj Digit Med 2024 (general); RADAR (motivates mitigation) | Shortcut learning in CXR is well established | The specific ablation protocol with an internal `mean` control and external re-evaluation is not something we found reported | not fetched; unverified | **unknown pending review** |
| OOD / abstention workflow | not searched | — | — | — | **unknown pending review** |
| Scanner-associated deferral disparity + workload accounting | not searched | — | Our measurement is specific: 20.1% vs 6.9% deferral, 7 TB referrals intercepted, +233 reviews | `OOD_GATE_AND_SCANNER_AUDIT.md` | **unknown pending review** |
| Low-compute inference on a 6 GB GPU | Domain-Aware, RADAR (both frozen-backbone) | Frozen backbones are the standard low-compute route | Nothing distinctive | — | **already established** |

## What this means for the existing write-ups

`report/paper_draft.md` and `docs/p2_focal_pooling_final.md` present focal pooling as the
central contribution. **That framing is not defensible** given RADAR, which does the same
thing more thoroughly and adds the invariance objectives we lack. Both documents need a
prior-art section and a softened claim. The *measurements* in them remain valid; the
*novelty* implied around them does not.

One claim in `p2_focal_pooling_final.md` is also now directly contradicted by prior art:
the suggestion that discarding patch tokens is an oversight "the whole field defaults to"
is wrong — RADAR and GLoRI both exploit them.

## Honest residual

Three things in this project were **not** found in the fetched prior art, and are the only
candidates for a contribution:

1. A frozen-backbone TB model evaluated against a **certified commercial comparator on
   identical patients** in a consecutive screening cohort (0.7242 vs 0.7862).
2. The **operational cost accounting** of an OOD gate in a screening workflow, including
   TB referrals intercepted and scanner-associated workload disparity.
3. The **monotone inverse relationship** observed across six interventions between TBX11K
   fit and external transfer.

All three are evaluation findings, not architecture. None has been checked against a
systematic literature search — only against the five sources above, two of which were not
retrievable. They should be treated as *unknown pending review*, not as novel.
