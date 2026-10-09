"""P3-09: decision D19, Grad-CAM configuration for the RAD-DINO classifier.

  1. python -m p3_xai.d19 criterion          -> commit decisions/p3_gradcam_config.json (before any testing)
  2. python -m p3_xai.sanity ...             -> fills value/evidence after the development sanity check
  D19 is locked only on P2's REAL classifier_module() (after D11), never on a stub or stand-in.
"""
from __future__ import annotations

import sys

from p3_det import decisions as dec

D19_PATH = dec.REPO / "decisions" / "p3_gradcam_config.json"

CRITERION = (
    "Grad-CAM configuration for the RAD-DINO classifier (methodology: Grad-CAM on the last RAD-DINO block with a "
    "reshape rule, TB target and crop-to-512 mapping), fixed a priori and checked on development only. "
    "(1) Target layer: the first LayerNorm (attention input, norm1) of the LAST transformer block of the "
    "classifier's ViT backbone, found by inspecting classifier_module() (last element of the backbone's block "
    "list). Rationale: the block output reaches a CLS-pooled head only through a per-token LayerNorm, so patch "
    "tokens there receive zero gradient; norm1 lets the gradient reach patch tokens through the last block's "
    "attention. (2) Reshape rule: drop the first P tokens, P = 1 (CLS) + num_register_tokens read from the "
    "backbone config; reshape the rest to [H/p, W/p], p = patch size from the config, H x W = classifier input "
    "size; fail if the token count disagrees. (3) Target: the TB logit of the classifier output (binary head: "
    "output 0; three-class head: the 'tb' output), before sigmoid/softmax. (4) Mapping: the classifier input is "
    "the lung crop of img_for_analysis prepared exactly as P2's classifier input; the Grad-CAM map "
    "(pytorch-grad-cam 1.5.7 GradCAM: ReLU, min-max scaled, resized to the input) is resized bilinearly to the "
    "crop size, min-max normalised to [0, 1] inside the crop, pasted into a zero 512x512 canvas at crop_box and "
    "returned as float16. Acceptance (model-randomisation sanity check, Adebayo et al. 2018): with the "
    "classifier head re-initialised (seed 0), the mean Spearman correlation between original and randomised "
    "heatmaps, over pixels inside the crop, on 20 development images (seeded sample, seed 0, 10 TB + 10 non-TB) "
    "must be < 0.5; the same check with head + last block re-initialised is reported. Peak GPU memory of one "
    "explain() call must be <= 5.5 GB on the RTX 4050."
)


def criterion_template(today: str | None = None) -> dict:
    return dec.new_decision(
        "D19", "Grad-CAM configuration (target layer, reshape rule, target class, crop->512 mapping)", CRITERION,
        ["last block norm1 (attention input)", "last block output"], ["development"], today)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["criterion"]:
        print("usage: python -m p3_xai.d19 criterion", file=sys.stderr)
        return 2
    dec.write_criterion(D19_PATH, criterion_template())
    print(f"wrote {D19_PATH.relative_to(dec.REPO)}; commit it before running the sanity check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
