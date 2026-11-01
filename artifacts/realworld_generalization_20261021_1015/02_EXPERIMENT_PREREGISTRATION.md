# Preregistration: does the training task definition explain the past-TB failure?

**Written before any arm was fitted and before any new score existed.** Hashed and recorded
in `tables/prereg_hash.txt` before `scripts/phase2_experiment.py` was run.

---

## 1. The observation that motivates this

From `tables/vietnam_negative_spectrum.json`, computed from committed per-case scores with
no fitting:

| group | n | median score | referral rate at the fixed threshold |
|---|---|---|---|
| Xpert-negative, **no** past TB | 1,578 | **−0.338** | 0.232 |
| Xpert-negative, **past TB** | **850** | **+2.175** | **0.458** |
| Xpert-positive | 280 | +3.123 | 0.621 |

**35.0% of the Vietnam negatives have a documented history of past tuberculosis**
(`past_tb`, defined in the Dryad README). Their scores sit much closer to the positives
than to the other negatives.

Decomposing discrimination by that variable:

| task | n | TB | ours | 95% CI | Lunit 3.9.0.1 |
|---|---|---|---|---|---|
| active TB vs negatives **without** past TB | 1,858 | 280 | **0.7811** | [0.7546, 0.8057] | 0.8170 |
| active TB vs negatives **with** past TB | 1,130 | 280 | **0.6182** | [0.5818, 0.6552] | 0.7192 |
| active TB vs all negatives | 2,708 | 280 | 0.7240 | [0.6971, 0.7490] | 0.7827 |

The commercial comparator degrades on the same axis (0.8170 → 0.7192, −0.098), so prior TB
is intrinsically hard. **We degrade about 1.7× as much** (−0.163).

## 2. Hypothesis

The training task is defined differently from the evaluation task, in two ways that both
push toward flagging inactive disease:

1. **`latent_tb` is labelled positive.** `artifacts/p2/labels_tbx11k_trainval_v1.csv`
   assigns `tb_label = 1` to `latent_tb` (139 images, 93 in train) and
   `active_latent_tb` (30, 21 in train), while its own `active_tb_label` column marks
   `latent_tb` as 0. The head is therefore trained to call radiographic signs of
   **inactive** TB positive — and then evaluated against Xpert Ultra, which calls exactly
   those participants **negative**.
2. **Negatives are healthy-only.** The current configuration trains on 2,700 `healthy`
   images and excludes all 2,700 `sick_non_tb` images, so the model never sees an abnormal
   chest labelled negative.

**Prediction.** Aligning the training definition with the Xpert reference standard — active
disease positive, inactive and non-TB-abnormal negative — should specifically improve
discrimination between active TB and past-TB negatives, the subgroup where the system
currently fails.

**Falsification.** If the primary endpoint does not improve by at least the minimum effect
below, or if it improves only by degrading the overall cohort, the hypothesis is not
supported by this experiment.

## 3. Arms

All arms: frozen RAD-DINO, lungs-only masking, **regional 12×12 max-pooling**, heads fitted
on **TBX11K train only**, early-stopped on TBX11K development, seeds **17, 29, 43, 61, 79**.
Identical optimiser, learning rate, weight decay, batch size and epoch budget. The only
variable is **which TBX11K rows are positive and which are negative**.

| arm | positives (train n) | negatives (train n) | total |
|---|---|---|---|
| **H** baseline, current | active_tb + latent_tb + active_latent_tb (541) | healthy (2,700) | 3,241 |
| **L** latent reassigned | active_tb + active_latent_tb (448) | healthy + latent_tb (2,793) | 3,241 |
| **S** sick negatives added | active_tb + latent_tb + active_latent_tb (541) | healthy + sick_non_tb (5,400) | 5,941 |
| **X** both changes | active_tb + active_latent_tb (448) | healthy + latent_tb + sick_non_tb (5,493) | 5,941 |

**Primary comparison: X against H.** L and S are prespecified secondary arms whose only
purpose is to attribute any X effect to one change or the other. No other arm will be
fitted, and the arm set will not be extended after seeing results.

Development AUROC for early stopping is computed under each arm's own label definition, so
every arm is stopped on the task it was trained for.

## 4. Primary endpoint and decision rule

**Primary endpoint:** AUROC for **active TB vs Xpert-negative participants with past TB**
on the Vietnam cohort (n = 1,130; 280 positives, 850 negatives). Baseline value **0.6182**.

**Minimum effect size: +0.05 AUROC.** Chosen before scoring, as roughly half the distance
from our 0.6182 to the commercial comparator's 0.7192 on the identical task. A smaller
movement is not worth a task-definition change.

**Decision rule — all three must hold for ACCEPT:**

1. Primary endpoint improves by **≥ +0.05** AUROC, arm X over arm H.
2. The **paired bootstrap 95% CI** for that difference (4,000 resamples, seed 0, identical
   cases) **excludes zero**.
3. **Guardrail:** AUROC on the full Vietnam cohort (active TB vs all negatives) does not
   fall by more than **0.02** relative to arm H.

Any other outcome is **REJECT** or **INCONCLUSIVE**, recorded as such. The rule will not be
reinterpreted after the result is seen.

**Secondary, reported but not decisive:** TBX11K development AUROC per arm; the full-cohort
Vietnam AUROC; discrimination against negatives *without* past TB (must not collapse);
referral rate and specificity at a threshold fixed on TBX11K development; and the L and S
attribution arms.

## 5. The honesty constraints this experiment operates under

- **This is exploratory and hypothesis-generating, not validation.** The hypothesis was
  generated by inspecting the full Vietnam cohort, so **neither the TUNE nor the REPORT
  half is clean for it**. Both halves are reported, and the pooled figure is the primary
  one precisely because splitting would imply a cleanliness that does not exist.
- **No Vietnam label is used to fit anything.** All heads are fitted on TBX11K train and
  early-stopped on TBX11K development. The Vietnam `past_tb` variable is used only to
  *define the evaluation subgroup*, not to select, weight or calibrate anything.
- **No invented metadata.** `past_tb` is documented in `data/raw/vietnam/README.md`;
  `latent_tb`, `active_tb` and `sick_non_tb` come from the TBX11K archive's own folder and
  annotation structure, already committed in `labels_tbx11k_trainval_v1.csv`.
- **TBX11K `internal_test` (901 cases) is not touched.** It remains the project's only
  unexposed partition and is deliberately preserved.
- **No threshold is chosen on Vietnam.** Operating points use the TBX11K development
  threshold at 70% specificity under each arm's own labels.
- **A positive result here would not be a generalisation claim.** It would be a reason to
  include the change in the locked configuration for the next screening cohort, which is
  where it could actually be tested.

## 6. What is not being claimed

Not a novel architecture: this is a label-definition change, the oldest idea in applied
machine learning. Not independent validation. Not clinical readiness. Not a causal proof —
even a positive result would be consistent with several mechanisms, since the two label
changes alter the positive class, the negative class and the class balance at once, which
is why the L and S arms exist.
