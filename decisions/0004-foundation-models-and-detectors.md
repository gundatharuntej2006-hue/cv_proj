# ADR 0004: Model Selection — RAD-DINO and D-FINE

## Status
Accepted

## Context
Previous TB models relied on ImageNet-pretrained ResNet-50 or heavy vision transformers that struggled to generalize across diverse clinical sites (Shenzhen vs Montgomery vs TBX11K) or were too heavy for edge deployment (<2s latency).

## Decision
1. **Classification:** We select **RAD-DINO** (Perez-Garcia et al., Nature Machine Intelligence 2025), a self-supervised foundation model pretrained on hundreds of thousands of chest radiographs, enabling strong out-of-the-box feature representations for 3-class classification (Healthy / Sick Non-TB / TB).
2. **Lesion Detection:** We select **D-FINE** (Peng et al., 2024), a lightweight distribution-refinement DETR architecture that offers superior localization precision for subtle consolidation and apical cavities at high inference frame rates.

## Consequences
- Dramatic reduction in annotation requirements.
- Generalization to unseen external datasets (Shenzhen, Montgomery).
- Meets the < 2.0s per image inference constraint.
