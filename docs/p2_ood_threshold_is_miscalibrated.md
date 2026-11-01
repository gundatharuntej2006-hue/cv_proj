# The OOD gate throws away sick patients for nothing

**Owner:** Person 2 · **Date:** 2026-10-14 · **Affects:** D13 (not yet locked), P2-11, the
demo bundle · **Status:** measured, with a recommended replacement rule

## How this surfaced

A real TB image from TBX11K, uploaded to the inference API, came back **rejected as
out-of-distribution** — score 0.5781 against the fitted threshold 0.5746. At α = 0.05 some
in-distribution rejection is expected, so the first question was whether those rejections
land evenly across classes. They do not.

## 1. The gate rejects abnormal chest X-rays, not just alien inputs

Flag rate by class at the fitted threshold (0.5746, bank of 4,000 training embeddings, k=5):

| class | train | development | calibration |
|---|---|---|---|
| healthy | 0.33% | 0.33% | 1.00% |
| sick_non_tb | 4.15% | **9.67%** | **12.25%** |
| active_tb | 2.11% | 4.26% | 8.97% |
| latent_tb | 3.23% | 0.00% | 11.11% |

**Diseased lungs are flagged roughly twelve times as often as healthy ones.** The mechanism
is not mysterious: the score is a k=5 cosine distance to neighbours, healthy chest X-rays
form a tight cluster, and disease makes an image sit further from everything. The "OOD"
score is in substantial part an abnormality score.

Checking the specific worry first — is it biased against TB? **No, not against TB
particularly**: on development, TB cases are flagged 3.39% [0.93%, 11.54%] versus 5.00%
[3.52%, 7.05%] for non-TB, which is indistinguishable. The bias is along
healthy-versus-abnormal, and TB sits in the middle of it.

That is still a safety problem, and a well-hidden one. **A rejected case carries no
probability**, by design. So every case the gate discards is absent from sensitivity,
specificity and AUC alike — the metrics are computed over the survivors. A gate that
preferentially discards abnormal images is therefore flattering every number in the report
while quietly removing the patients who most need the screen.

## 2. The rejections buy nothing

The gate exists to reject inputs that are not a frontal chest X-ray — a photograph, a
lateral view. So the real question is what the threshold trades off. Scoring both alien
pools (247 non-medical images, 501 CheXpert laterals) against the development frontals:

| threshold | non-medical caught | laterals caught | sick_non_tb wrongly rejected | TB wrongly rejected | healthy wrongly rejected |
|---|---|---|---|---|---|
| **0.5746** (fitted) | 100% | 100% | **9.67%** | **3.39%** | 0.33% |
| 0.60 | 100% | 100% | 6.00% | 1.69% | 0.00% |
| 0.65 | 100% | 100% | 3.00% | 1.69% | 0.00% |
| **0.70** | **100%** | **100%** | **1.00%** | **0.00%** | 0.00% |
| 0.725 | 100% | 100% | 0.33% | 0.00% | 0.00% |
| 0.75 | 100% | 99.8% | 0.33% | 0.00% | 0.00% |
| 0.80 | 100% | 99.0% | 0.00% | 0.00% | 0.00% |

**Alien detection is saturated at 100% across the whole useful range.** Moving the threshold
from 0.5746 to 0.70 costs *nothing at all* in alien detection and takes the wrongful
rejection of sick patients from 9.67% to 1.00%, and of TB cases from 3.39% to zero.

The fitted threshold is simply in the wrong place. There is a wide margin between where
valid chest X-rays stop and where aliens begin, and the conformal fit put the threshold at
the bottom of that gap instead of in the middle of it.

## 3. Why the method produced this

Not a coding error — a misuse of the right tool. The conformal procedure at α = 0.05
answers:

> give me a threshold whose false-flag rate on in-distribution data is **at most** 5%.

Because it places the threshold at the empirical 95th percentile of the development scores,
it does not merely permit a 5% false-flag rate, it **produces one by construction**,
whether or not any alien input sits anywhere near that boundary. An upper bound was being
used as a target.

That is exactly backwards for this gate. The guarantee we want is about the *aliens*, and
the in-distribution false-flag rate should be as low as alien detection allows — which here
is essentially zero.

## 4. Recommended amendment to D13

Replace the threshold-selection rule. Instead of "the conformal α = 0.05 threshold on
development", use:

> **The largest threshold that still rejects 100% of every known alien pool, less a safety
> margin.** Report the achieved in-distribution false-flag rate per class, and report the
> alien-detection rate at the chosen threshold.

On the current measurement that gives a knee at about 0.725–0.74 (laterals start slipping
through at 0.75), so the recommendation is **0.70**: a margin below the knee, 100% alien
detection on both pools, 1.00% false flags on sick frontals and 0.00% on TB.

Three things to keep from the original design:

1. **Keep the conformal machinery for the cross-site case.** `p2_ood_gate_finding.md` shows
   the threshold does not transfer between sites and that ~25 local images restore it
   conformally. That use is correct — there, the question genuinely is "bound the false-flag
   rate at a new site". The abstention rule below n = 1/α − 1 stays as it is.
2. **Keep reporting the per-class false-flag table.** It is the only thing that would have
   caught this, and it belongs in the report whatever threshold is chosen.
3. **Keep the threshold on the conservative side of the knee.** The two alien pools here are
   photographs and laterals; a real deployment will meet stranger inputs, and the margin is
   what covers them.

## 5. Caveats, and what this does not fix

- Two alien pools only, both of which are *easy* — a photograph of a cat and a lateral chest
  film are both far from a frontal CXR. A harder alien (a paediatric film, a different
  modality, a heavily annotated image) might sit inside the margin, which is the argument
  for not pushing the threshold all the way to the knee.
- 59 TB cases in development, so "0.00% of TB wrongly rejected" means 0 of 59 — the upper
  confidence bound is about 6%, not zero.
- Provisional split; D13 is **not locked**, so this costs nothing to change now.
- This does **not** fix the cross-site failure. At a new hospital the whole score
  distribution shifts (`p2_ood_gate_finding.md`), and no choice of a single global threshold
  addresses that. The two findings are independent: this one says the threshold is in the
  wrong place *on the site it was fitted for*.

Evidence: `artifacts/p2/dryrun/ood_class_bias.json`, reproduced by
`scripts/p2_11_ood_class_bias.py`.
