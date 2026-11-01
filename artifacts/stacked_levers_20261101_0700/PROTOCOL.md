# Protocol — the three external levers, stacked

**Hashed before fitting.** The script re-hashes this file and refuses to run if changed.
**Exploratory.** TBX11K calibration is the primary evaluation split; Vietnam TUNE is
exposed and reported for context only. Nothing is adopted into production.

## 1. Hypothesis — and its provenance, stated first

Three independent training-side choices have each been measured as "invisible on
TBX11K calibration, positive on the Vietnam screening cohort":

| lever | run | calibration Δ | TUNE Δ |
|---|---|---|---|
| active-TB target (latent → negative) | R12 / R17 | −0.0037 | +0.0203 [+0.0127, +0.0279] |
| three training sources (+ Shenzhen, Montgomery) | R19 | +0.0033 | +0.0036 (tb) / +0.0077 (active) |
| RAD-DINO-MAIRA-2 encoder | R20 | −0.0004 | +0.0188 [+0.0015, +0.0364] |

> **Hypothesis:** the three levers are additive. A head using all three should improve
> on the production head **in either direction**, and on TUNE should land near 0.77.

**Every lever was identified on TUNE.** TUNE is therefore not an evaluation set for the
stack; it is where the stack was designed. The primary endpoint is calibration, the only
split none of the levers was chosen on. The TUNE number is reported as context and as
a *prediction* (≈ 0.77 if additive), not as validation. Vietnam REPORT stays sealed.

## 2. Arms — one fitting path

Trainer identical to production and to R17 / R19 / R20: `Linear(768→1)` per region,
max over 144 regions, AdamW 3e-3 / batch 256 / wd 1e-4 / ≤120 epochs, early stopping
on TBX11K development image-level AUROC (patience 15), logit adjustment, seeds
17 / 29 / 43 / 61 / 79, seed-mean logits. NLM never enters the development set.

| arm | encoder | target | training rows |
|---|---|---|---|
| **A** | RAD-DINO | tb | TBX11K 3,241 — must reproduce 0.9498 |
| **MB1** | RAD-DINO | active | + 800 NLM — must reproduce 0.9497 (R19) |
| **X** | **MAIRA-2** | **active** | **+ 800 NLM** — the stack |
| X_noNLM | MAIRA-2 | active | TBX11K 3,241 — ablation: encoder + label |
| X_tb | MAIRA-2 | tb | + 800 NLM — ablation: encoder + sources |

MAIRA-2 features for TBX11K train / development / calibration and Vietnam TUNE are the
R20 caches (`a6132efb…`); the 800 NLM images are embedded here with the R20 embedder from
the same lungs-masked PNGs as the RAD-DINO NLM cache. Under the active target every NLM
TB image is treated as active (R19 assumption, unchanged).

## 3. Evaluation

**Primary:** paired ΔAUROC on **TBX11K calibration** (899; 81 active-TB positives),
**X minus A**, scored against `active_tb_label`, paired bootstrap 4,000 resamples seed 0,
identical indices. Minimum meaningful effect **+0.02**, interval excluding zero.
Half-width > 0.05 → INCONCLUSIVE.

**Secondary:** X − MB1 (what the encoder adds on top of the other two), the two
ablations, every arm on TUNE with CI, sensitivity at the development-fitted
0.70- and 0.90-specificity thresholds, AUPRC. Additivity check: X − A on TUNE against
the sum of the three single-lever deltas (+0.047).

**Decision rule.** ACCEPT only if the primary contrast meets its bar. Whatever TUNE
shows, the verdict on adoption is unchanged by it.

## 4. Partitions

TBX11K `internal_test` (901) and Vietnam REPORT (1,356) are not read, not embedded. No
per-case output. No Vietnam patient identifier is written to any table.
