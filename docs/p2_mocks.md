# P2-02 — mocks and R2 stubs, for Persons 1, 3 and 4

**Owner:** Person 2 · **Task:** P2-02 · **Status:** delivered, Week 1 · **Regenerate:** `make mocks-p2`

Person 2 publishes five contracts. Real versions of all five arrive between Weeks 3 and 8.
Until then these mocks stand in, at the same relative paths under `mock/`, selected with
`TB_MOCK=1`. Everything is deterministic given `--seed` (default 20260929).

| Contract | Mock path under `mock/` | Rows | Real version |
|---|---|---|---|
| C03 OOD collection | `data/splits/ood_heldout_v1.csv` | 180 | P2-10, W5 |
| C07 embeddings | `artifacts/p2/emb/radino_tbx11k_{crop_m0.10_clahe0,full_clahe0}_v1.npz` | 8,400 × 768 | P2-03, W1–2 |
| C08 scores | `artifacts/p2/scores/scores_tbx11k_{development,calibration,internal_test}_v1.parquet` | 4,200 / 7,350 / 7,350 | P2-07 onward, W4–8 |
| C09 gate | `artifacts/p2/gate/gate_tbx11k_{development,calibration}_v1.parquet`, `gate_ood_ood_eval_v1.parquet` | 840 / 1,470 / 60 | P2-05 + P2-11, W3–6 |
| C17 dedup | `artifacts/p2/dedup/near_dup_candidates_v1.csv` | 4 pairs | P2-04, W2 |

`mock/mock_manifest_p2.json` records the seed, every path and the answer key below.
Total size 24 MB, gitignored.

## What is in the mocks on purpose

**The scores have a known right answer (for P4-11).** Latent ability is drawn N(+1,1)
for TB images and N(−1,1) for the rest. The emitted logit is 1.7 × the exact posterior
log-odds for that split, so the scores are overconfident by a factor of 1.7 and
temperature scaling has something real to correct.

Assert against `known_answers.recovered_temperature`, not against 1.7. The generator
fits the temperature on the files it just wrote and records it per split and per model.
Each split is a finite draw, so the value that actually minimises the NLL drifts from
1.7 by up to about 0.2 — on these files, development sits near 1.85 and internal test
near 1.70. All four binary models within one split agree closely, which is the tell that
the drift belongs to the split's draw and not to the model. Primary-model AUC is about
0.87 to 0.91 by split; EfficientNet-B3 is deliberately a little worse.

**The mocks agree with each other.** C17 is not a list of invented pairs; it is a real
all-pairs cosine search over the C07 mock, into which four duplicates were planted. C09's
threshold is a real order statistic of the mock development scores, so the D13 criterion
(development pass rate 95.0 ± 0.2 %) holds on the mock itself — if your code reports
94.0 % you have a bug, not a bad mock.

**The awkward cases are present.** A train ↔ internal_test duplicate (the H1 leak dedup
exists to catch); a pair whose cosine is exactly τ = 0.98, so **P1 must decide `>=`
against `>` now** rather than in Week 8; a model row with a null `margin` (the full-image
variant); quality failures carrying reasons from the C09 vocabulary; and an OOD set that
mostly fails the gate but not entirely.

**The C03 sources are fake on purpose.** Every row reads `mock_*_source` under the
licence string `MOCK-NOT-A-LICENCE`, so a mock row can never be pasted into the committed
file or mistaken for a real source. Real OOD sources and their licences are chosen in
P2-10 and recorded in D15 during Week 5.

## R2 stubs

```python
from p2_gate.api import gate                       # -> GateResult
from p2_cls.api import classify, classifier_module # -> ClsResult, torch.nn.Module
```

Signatures are frozen at `contracts-v1`. The weights are not: the backbone is the real
pinned RAD-DINO, the head is randomly initialised, and **every result carries
`is_stub=True`**. The P4-09 gate must refuse any reported number carrying that flag.

Pass `pretrained=False` to build the same architecture with random weights — CI then runs
the shape and gradient tests without pulling a 330 MB checkpoint.

### For Person 3, Grad-CAM (P3-09) — two things that cost a day each

1. **A frozen backbone produces activations with no gradient.** With every parameter at
   `requires_grad=False` and an ordinary input, the last block's output is a leaf with
   `requires_grad=False`: `retain_grad()` raises, and if that exception is swallowed the
   heatmap comes out all zeros with no error. `classifier_module()` handles this itself in
   a forward hook on `target_layer`, so you do not have to remember. Measured in P2-01:
   forward plus backward peaks at **0.84 GB**, against the 5.5 GB budget.
2. **ViT tokens are not a picture.** `last_hidden_state` is [B, 1370, 768] for a 518-pixel
   input at patch 14. Use `module.reshape_transform(tokens)` for [B, 768, 37, 37]. It
   computes the number of prefix tokens from the config instead of assuming "drop one", so
   a revision with register tokens would not silently shift every heatmap by a patch.

`module.preprocess(img)` is the resize path P2 uses. Use it rather than your own, or the
heatmap will not line up with the boxes.

### For Person 4, `run_case` (P4-10)

`gate()` really does measure the pixels, so a blank, an over-exposed and a heavily cropped
image are genuinely rejected with a reason from the C09 vocabulary — the "gate rejected"
fixture behaves like the real thing. The OOD score, by contrast, is a deterministic
function of image content and **is not a distance to anything**; it only populates the
field until the real bank exists (P2-11, W6).

## Tests

`make test-p2` — 36 tests covering contract schemas and dtypes, the C08 invariants Parquet
cannot express, agreement between C17 and C07, the D13 pass rate, the stub's
forward/backward path and the token reshape. Add `-m "not slow"` to skip the one test
needing real weights.

## Two notes for the Week 1 contract sign-off

- **C08 `logits_3c` is a variable-length list, not `fixed_size_list(float32, 3)`.**
  Parquet has no fixed-size list type: a null in such a column comes back as an empty
  list and the file will not re-read at all. The length-3 rule is enforced by
  `p2_cls.contracts.check_c08_invariants` instead. Please match this in `tbcore`.
- **Person 1:** the pair at exactly τ needs a `>=` or `>` ruling in D06 before splits
  freeze in Week 2.
