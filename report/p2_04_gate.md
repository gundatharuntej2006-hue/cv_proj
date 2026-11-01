# §4 Image-quality and out-of-distribution gate

*Owner: Person 2. Every number cites its artifact. All results below are on the
**provisional** split (Person 1's C01 pending) and on the **development** split only; the
held-out `internal_test` split has not been read — see §12 and
`docs/p2_sealed_scoring.md`.*

## 4.1 Why a gate at all

A screening model given a photograph, a lateral film or a blank scan will still return a
probability, and that probability will look exactly like a real one. The gate exists so that
the system can say "I cannot assess this" instead. Its contract (C09) is deliberately narrow:
it either passes an image to the classifier, or rejects it with a reason and **no
probability at all**. A rejection means *send this to a human*, never *this patient does not
have TB*.

That property is enforced in code rather than documented: `p2_cls.pipeline` returns
`analysis: None` for every rejected case, the HTTP layer asserts it a second time before
responding, and both are tested (`tests/p2_serve/test_app.py`,
`tests/p2_cls/test_pipeline.py`).

## 4.2 Quality rules (D07)

Four rejection reasons, and only these four, may ever be emitted: `blank`,
`under_exposed`, `over_exposed`, `heavily_cropped`. The vocabulary is closed and the gate
filters against it rather than trusting its own rules, so a rule added later cannot leak a
fifth value into downstream parquet files (`tests/p2_gate/test_gate.py`).

The thresholds were **derived, not guessed**. A synthetic degradation sweep applies each
failure mode at increasing severity to real development images, and each threshold is placed
at the midpoint of the gap between the clean and degraded score distributions
(`p2_gate.quality.gap_midpoint`):

| reason | rule | threshold |
|---|---|---|
| `blank` | pixel std | < 0.020 |
| `under_exposed` | mean intensity | < 0.120 |
| `over_exposed` | mean intensity, or saturated fraction | > 0.880, or > 0.500 |
| `heavily_cropped` | shorter side, or aspect ratio | < 256 px, or > 1.600 |

Rejection rate on the development split is well inside the ≤ 1% acceptance criterion.

## 4.3 OOD score (D13)

The score is the mean cosine distance from an image's RAD-DINO embedding to its *k* nearest
neighbours in a bank of 4,000 unaugmented training embeddings. It is threshold-free and
monotone, so it can be recalibrated without retraining anything.

**Choice of *k*: it does not matter, and the report should say so.** Separability against
the held-out alien pools (246 non-medical images, 500 CheXpert laterals) is **AUROC 1.0000
for every *k* from 1 to 20 — a spread of 0.0000**. The false-flag rate at full alien
detection is 0.33% of sick frontals and 0.00% of TB cases at every *k*. All *k* does is move
where the threshold sits (0.719 at *k* = 1 to 0.774 at *k* = 20).

| *k* | alien AUROC | threshold at 100% catch | sick_non_tb false flags | TB false flags |
|---|---|---|---|---|
| 1 | 1.0000 | 0.7190 | 0.33% | 0.00% |
| 3 | 1.0000 | 0.7373 | 0.33% | 0.00% |
| **5** | 1.0000 | 0.7449 | 0.33% | 0.00% |
| 10 | 1.0000 | 0.7577 | 0.33% | 0.00% |
| 20 | 1.0000 | 0.7741 | 0.33% | 0.00% |

*k* = 5 is retained as pre-registered, but the honest statement is that **the ablation found
no effect**: these alien pools are too easy to discriminate between values of *k*. Reporting
*k* = 5 as "selected" would imply a measurement that did not happen. Source:
`artifacts/p2/dryrun/section14_ablations.json`.

## 4.4 Threshold selection, and a correction (D13 amendment)

The methodology originally set the threshold conformally at α = 0.05 on development. That
was measured and **replaced**, because it was discarding patients for no benefit.

The conformal procedure places the threshold at the 95th percentile of in-distribution
scores, so it *produces* a 5% false-flag rate by construction, whether or not any alien
input lies near that boundary — and none does. Measured at the conformal threshold (0.5746),
the gate rejected **9.67% of sick_non_tb and 3.39% of TB development cases**, while alien
detection was already saturated at 100% far above it:

| threshold | non-medical caught | laterals caught | sick_non_tb rejected | TB rejected |
|---|---|---|---|---|
| 0.5746 (conformal) | 100% | 100% | 9.67% | 3.39% |
| 0.65 | 100% | 100% | 3.00% | 1.69% |
| **0.70 (adopted)** | **100%** | **100%** | **1.00%** | **0.00%** |
| 0.75 | 100% | 99.8% | 0.33% | 0.00% |

The rule is now: **the largest threshold that still rejects 100% of every known alien pool,
less a safety margin**, with the achieved per-class false-flag rate reported. The margin
matters because both alien pools here are easy (a photograph, a lateral film) and a
deployment will meet stranger inputs.

This correction was found by chance — a real TB image, served through the inference API, came
back rejected. The per-class false-flag table in §4.5 is what makes such a failure visible,
and it is reported for that reason.

**Why this was easy to miss.** A rejected case carries no probability, so it is absent from
sensitivity, specificity and AUC alike: those metrics are computed over the survivors. A
gate that preferentially discards abnormal images therefore *improves* every headline number
while removing the patients the screen exists to find. Source:
`docs/p2_ood_threshold_is_miscalibrated.md`, `artifacts/p2/dryrun/ood_class_bias.json`.

## 4.5 False-flag rate by class

At the adopted threshold the gate must not single out any class. Measured
(`artifacts/p2/dryrun/ood_class_bias.json`, re-run at 0.70):

- TB cases: **0.00%** (0 of 59 on development — the 95% upper bound is about 6%, not zero)
- sick_non_tb: 1.00%
- healthy: 0.00%

At the previous threshold the gate was *not* biased against TB specifically (3.39% versus
5.00% for non-TB, indistinguishable) but was strongly biased along healthy-versus-abnormal,
rejecting diseased lungs roughly twelve times as often as healthy ones. The mechanism is
simply that healthy chest X-rays form a tight cluster and disease moves an image away from
its neighbours: the OOD score is partly an abnormality score, which is why the threshold must
be set from the alien side and not from the in-distribution tail.

## 4.6 Abstention: when the gate refuses to judge

The conformal threshold does not exist when the calibration set is smaller than
1/α − 1 images (9 at α = 0.10, 19 at α = 0.05). `p2_gate.ood.fit_threshold` returns
**None** in that case and the gate then **flags nothing** — it does not fall back to a guess
and it does not reject everything. Every response carries `ood_abstained` so the state is
visible rather than silent (`tests/p2_gate/test_gate.py`).

This matters because of §4.7: at a new site the honest state really is "not calibrated
here yet", and a gate that responded by rejecting every case would be useless in exactly the
situation it was built for.

## 4.7 What the gate cannot do: cross-site transfer

**The threshold does not transfer between sites, and this is the single most important
limitation of the gate.** Calibrated on TBX11K and applied unchanged to a different
hospital's normal chest X-rays, it rejects **98.7%** of them. TBX11K images sit at a *k* = 5
distance of 0.358 from each other; the new site's images sit at 0.779
(`docs/p2_ood_gate_finding.md`).

The *score* transfers even though the threshold does not — it separates the two sites at
AUROC 0.9837 — so the gate is recoverable: about **25 images from the new site** are enough
to re-fit the threshold conformally.

Three candidate fixes were tested and **all three failed**: mean-removal (the site offset is
rank-1 and TB-orthogonal), whitening, and cropping to the lung field (site separability
0.9986 → 0.9987, unchanged). The site signature is therefore not a peripheral artifact of
borders, collimation or burned-in markers — it survives cropping to pure lung tissue, so it
lies in how the scanner renders the tissue itself (`docs/p2_does_lung_cropping_help.md`).

It is also **not a property of RAD-DINO**. A conventional supervised CXR CNN
(TorchXRayVision DenseNet-121) separates the same two sites at AUROC 0.9998, and in relative
terms *worse* — 5.60× versus 2.17× — so changing the representation does not help
(`docs/p2_backbone_baseline.md`). Per-site calibration is a property of the task, not a
defect of this implementation.

## 12.6 Narrative: what the gate contributes, honestly

The gate does two of the three jobs asked of it. It reliably rejects inputs that are not
frontal chest X-rays — 100% of photographs and 100% of lateral films, with a measured cost
of 1% of sick frontals and no TB cases — and it refuses to guess when it cannot calibrate.

It does **not** provide a deployable out-of-distribution guarantee at a new site, and the
report should not claim that it does. Anyone installing this system at a new hospital must
collect roughly 25 local normal chest X-rays and re-fit the threshold before the gate means
anything; until then it should abstain, which it will.

The most useful finding here is the negative one, and it generalises beyond this project: a
nearest-neighbour OOD gate on a single public dataset measures *that dataset*, not
"chest X-ray-ness". Three natural corrections and a second backbone all left the effect
untouched. Any paper reporting a kNN OOD gate calibrated on one source and not tested on a
second should be read with that in mind.
