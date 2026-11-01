# Slide deck — 15-minute talk

*Owner: Person 2 (P2-16). Content draft; formatting decided at assembly. Numbers provisional.*

**Structure.** Four minutes setting up, eight on the finding, three on what it means. The
deck deliberately spends more time on why the headline number is wrong than on the system,
because that is where the work is.

**Speaker's rule for this talk:** never show 0.9949 without 0.6151 on the same slide. The
moment they are separated, the talk becomes a different and less honest talk.

---

## 1 — Title

**Explainable TB Screening from Chest X-Rays with Lesion Localisation for Triage**
BAI506 Mini Project · Department of AI & ML, BMSIT&M · 2026–27
Tharun Tej · Atul Dhull · Ritika · Kunal

---

## 2 — Why tuberculosis, why radiography

- ~1.25 million deaths a year; leading infectious killer outside pandemic years
- Case-finding is the bottleneck — a large share of cases are never diagnosed
- WHO recommends chest X-ray as a **triage** test before molecular confirmation: cheap,
  fast, available where Xpert cartridges are not
- So: can a model do the triage?

*Say: the literature suggests yes. That is what we set out to check.*

---

## 3 — What we built

Pipeline diagram, left to right:

`image → quality + OOD gate → classifier → lesion boxes + heatmap → calibrated triage → worklist`

- Gate **abstains** rather than guessing
- Frozen RAD-DINO backbone + linear head — runs on a 6 GB laptop
- Held-out data read **exactly once**, enforced in code: write-once seals, signed single-use
  tokens held by someone else, append-only ledger
- Four workstreams, one contract per producer

---

## 4 — It works, and it scores 0.9949

| | |
|---|---|
| TBX11K development AUC | **0.9949** |
| published state of the art (SymFormer) | 0.9910 |

*Pause here. Let it sit. Then:* **"We spent the rest of the project finding out what that
number is made of."**

---

## 5 — The question: what is the model looking at?

Show three images side by side: full, lungs-only, border-only (the masked frame).

**"We deleted the lungs and asked it again."**

---

## 6 — The answer

| what the model can see | AUC |
|---|---|
| everything | 0.9949 |
| lung field only | 0.9889 |
| everything except the lungs | 0.9532 |
| **outer 12% frame only** | **0.9094** |

**The border of the image predicts tuberculosis at 0.9094** — shoulders, collimation,
background. No lung tissue.

---

## 7 — Ruling out the obvious objection

*"The segmenter trims the apices, and TB is an upper-lobe disease — you left pathology in."*

- Dilate the erased region by 48 px to remove the apices: costs **0.0039**
- `border_only` needs no segmentation at all — fixed rectangle, every image
- Flip the image left-right: **ΔAUC +0.0001**, though the thorax is strongly asymmetric
- Lungs-only is *significantly worse* than the full image

Four independent checks, same direction.

---

## 8 — Why: how the benchmark was built

TBX11K collected cases and controls as **separate groups**. Anything that differs
systematically between those groups is free signal — and the model used it.

*Not a bug in our code. The structure of the data, learned faithfully.*

---

## 9 — Does the shortcut travel?

| | TBX11K | Shenzhen | Montgomery | India | Vietnam |
|---|---|---|---|---|---|
| border only | **0.9094** | 0.5209 | 0.6082 | 0.6240 | 0.5474 |
| lungs only | 0.9889 | 0.9284 | 0.8554 | 0.7791 | 0.6430 |

**No.** At Shenzhen the border model is at chance. Lungs-only tracks the full image within
0.015 everywhere.

*This is how we know the external numbers are real — the obvious objection is answered with
a measurement, not an argument.*

---

## 10 — So trust the external number. Which one?

| site | type | AUC |
|---|---|---|
| TBX11K | benchmark | 0.9949 |
| Shenzhen, China | curated case-control | 0.9382 |
| Montgomery, USA | curated case-control | 0.8183 |
| DA+DB, India | curated case-control | 0.7932 |
| **Vietnam** | **consecutive screening cohort** | **0.6151** |

**"External validation" is not one thing.** The sets the field uses are near the easy end.

---

## 11 — The number that matters

**0.6151** on 2,708 consecutive screening attendees, Xpert Ultra on every one.

**0.7827** — a certified commercial product, *same patients, same images*.

| at 90% specificity | we detect | it detects |
|---|---|---|
| | **12%** | **45%** |

*The task is attainable at 0.78 there. We are not good enough for it.*

---

## 12 — Why we fail

| median score | cases | non-cases |
|---|---|---|
| ours | 0.953 | **0.886** |
| commercial | 0.889 | 0.568 |

**We call almost everyone positive.**

On the benchmark: TB-vs-healthy = **1.0000**, TB-vs-sick = 0.9861.
A screening queue is all "sick" — healed TB, COPD, chronic change.

**We trained a model to spot "abnormal" and deployed it where abnormal is the base rate.**

---

## 13 — The safety net doesn't catch it

| site | OOD distance | gate rejects | AUC |
|---|---|---|---|
| Shenzhen | 0.5331 | 1.2% | 0.9382 |
| **Vietnam** | **0.5312** | **0.8%** | **0.6151** |

Same measured distance. 0.32 AUC apart. The gate passes 99.2% of the cohort we fail on.

**Distance-based OOD gates detect appearance shift, not task shift.**

*We had claimed the opposite on three sites. The fourth falsified our own claim — say this
out loud, it is the most scientifically respectable moment in the talk.*

---

## 14 — Three things we'd tell anyone doing this

1. **Ablate the anatomy.** Mask it, re-run, report. Minutes on cached embeddings. Then run
   the same ablation externally — that separates a dataset artifact from a real signal.
2. **One consecutive cohort beats three case-control sets.** If you don't have one, say so as
   prominently as you say your AUC.
3. **Compare against something real.** "0.6151" is ambiguous. "0.6151 against 0.7827 on
   identical images" is a diagnosis.

---

## 15 — What this system is, honestly

**Not deployable.** On the population it would actually meet, it finds 12% of tuberculosis at
a usable specificity. We would not install it.

**Well characterised.** We can say what the headline number is made of, how much survives a
change of hospital, where it fails, and why.

The commercial comparator shows the task is doable — **the gap is ours to close, and we know
where to start.**

---

## 16 — Backup slides (for questions)

- **"Did you try fine-tuning?"** No — frozen backbone was a deliberate design choice for
  reproducibility and 6 GB. P2-17 is scoped; the commercial comparator suggests capacity may
  be the limit.
- **Latent vs active:** indistinguishable (AUC 0.55). 21% of positive calls are healed
  disease.
- **Robustness:** noise drops specificity 95%→26% while AUC only falls to 0.966. AUC is blind
  to a uniform score shift.
- **Backbone choice:** RAD-DINO beats DenseNet-121 by ΔAUC 0.0409, CI excluding zero.
  Crop margin, k, head type: no measurable difference.
- **Benchmark hygiene:** 158 cross-split duplicate pairs; 576 non-benchmark images reachable
  via `all_*.txt`; 12.1% of the held-out test split is relabelled public data.
- **Why not more data?** The deficiency is in the training *negatives*. Training on an
  external set destroys it as a test.
- **"Isn't 0.9949 still impressive?"** It is a benchmark number. The same model gets 0.6151
  where it would be used. That gap is the result.
