# What TBX11K actually contains — corrected, and why it unlocks external validation

**Owner:** Person 2 · **Date:** 2026-10-16 · **Supersedes the wording of:**
`p2_duplicate_finding.md`, `p2_external_site_landscape.md`, `p2_ood_gate_finding.md` §
"invalid control" · **Status:** verified empirically, two independent methods

## The correction

I previously wrote that "Shenzhen and Montgomery are inside TBX11K". That wording is
**wrong**, and the error matters because it led the project to write off its best available
external test set. The precise position:

| statement | true? |
|---|---|
| TBX11K's **benchmark** (11,200 images: 5,000 healthy, 5,000 sick non-TB, 1,200 TB) contains Shenzhen/Montgomery | **No.** It is all self-collected Chinese hospital data, one image per individual, culture-confirmed. The TPAMI paper (arXiv:2307.02848 §3.1.2) is accurate. |
| The **distributed archive** ships Shenzhen/Montgomery images | **Yes.** 400 of them, under `imgs/extra/mc+shenzhen/` (327 `CHNCXR_*`, 73 `MCUCXR_*`), plus 176 DA/DB images under `imgs/extra/da+db/`. |
| Those images enter training if you use the shipped lists | **Yes, via `all_*.txt` only.** `all_trainval.txt` has 8,976 entries against `TBX11K_trainval.txt`'s 8,400; the 576 extra entries are exactly the `extra/` directory. |
| **This project's** training set contains them | **No** — it uses the core `TBX11K_trainval.txt`. Verified, see below. |

So the operational advice in the original finding was right — *use the core lists or external
images enter training* — while the sentence describing the benchmark was not. Both halves
needed stating separately, and only one was.

## How the earlier measurement and today's measurement are both correct

The original check found **800 of 800** NLM images had a twin at cosine 0.9994; today's check
found **zero** pairs above 0.90. Not a contradiction — different reference sets:

- the original compared against the **whole archive on disk**, which includes `imgs/extra/`
  and `imgs/test/`;
- today's compared against the **8,400 core-list training embeddings**, which include
  neither.

Locating every twin by perceptual hash resolves it completely:

| TBX11K folder | images | NLM images with a near-identical twin |
|---|---|---|
| `imgs/extra/mc+shenzhen/` | 400 | **400** |
| `imgs/test/` | 3,302 | **400** |
| `imgs/health/` | 3,800 | 2 → **false positives** (see below) |
| `imgs/sick/` | 3,800 | 0 |
| `imgs/tb/` | 800 | 0 |

All 800 accounted for: 400 in `extra/`, 400 in the test split.

**The two `health` hits are hash artifacts, not duplicates.** A 16×16 average hash collides
readily on low-contrast normal chest films. Checked against RAD-DINO cosine, the actual
similarities are **0.2022** (`CHNCXR_0226_0` vs `h0299`) and **0.1649** (`CHNCXR_0234_0` vs
`h2841`) — unrelated images. Worth recording as a method note: a crude perceptual hash is
fine for *locating* candidates but must not be used to *confirm* them.

## Consequence 1: this project can do external validation today

Shenzhen (336 TB / 326 normal) and Montgomery (58 TB / 80 normal) are **genuinely unseen by
this project's models**, already on disk, and each carries its own controls from the same
hospital and screening programme. That last point is what makes them usable: a test set whose
positives and negatives come from different sources can be scored highly by recognising the
acquisition pipeline rather than the disease, which is the failure mode that disqualifies the
widely used Kaggle "Tuberculosis (TB) Chest X-ray Database" (its 3,500 positives come from
NLM/Belarus/NIAID, its 3,500 normals from the RSNA pneumonia challenge).

Evidence that they are unseen, two ways:

- **Embedding cosine:** max similarity between any of the 800 NLM images and any of the 8,400
  training embeddings is **0.8389**; zero pairs above 0.90; median 0.5329.
- **List membership:** none of the 576 `extra/` entries appears in `TBX11K_trainval.txt`.

Every "single-source only" caveat in the project's documents was therefore avoidable. Run at
`scripts/p2_22_external_validation.py`.

## Consequence 2: published TBX11K test numbers are ~12% NLM data

**400 of the 3,302 images in TBX11K's official test split (12.1%) are Shenzhen or Montgomery
radiographs**, not Chinese hospital data. The test images are distributed anonymised as
`unknown_####.png`, so this is not visible by inspection — it took a perceptual-hash search
to find.

Two implications for §17's research comparison:

1. Published test-split results (SymFormer at 99.1% AUC and the rest) are measured on a set
   that is one-eighth external data. That is not a flaw in those papers — they used the
   benchmark as distributed — but it does mean the benchmark's "test" split is not the
   single-source set its description implies.
2. If this project reports Shenzhen as external validation *and* compares against published
   TBX11K test numbers, there is overlap between our external set and their test set. It does
   not invalidate our number — our model never saw those images — but it must be stated when
   the two are put side by side.

## Consequence 3: a leakage trap for anyone using the shipped lists

Training on `all_train.txt` puts Shenzhen, Montgomery, DA and DB images into training **and**
leaves ~400 NLM images in the test split, so the same sources appear on both sides. Anyone
reproducing TBX11K work should check which list file they used. This is the substance of the
original finding and it stands unchanged.

## Method note for the report

This is the second time in this project that a confidently stated dataset-composition claim
turned out to be wrong in one direction or the other — first the claim that the NLM sets were
part of the benchmark, then a research summary asserting the opposite from the paper's prose
alone. Both were settled in minutes by measuring the data on disk.

The general rule earned here: **dataset provenance claims get verified against the bytes, not
against the paper, the README or a secondary source** — and a positive hash match gets
confirmed with a stronger representation before it is believed.

Evidence: `artifacts/p2/external/external_validation.json` (leakage block),
`scripts/p2_22_external_validation.py`, and the list-membership and hash checks reproduced in
this document's commit message.
