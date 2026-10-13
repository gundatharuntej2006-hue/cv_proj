# TBX11K duplicates, and: the NLM external sets are inside TBX11K

**Owner:** Person 2 · **Tasks:** P2-03, P2-04 · **Date:** 2026-10-06 · **Status:** verified
**Blocks:** the split freeze (H1, P1-07) · **Decision:** D06

## What was found

Embedding all 12,278 TBX11K images with RAD-DINO and running an all-pairs cosine search
over the full-image variant returns **158 cross-split pairs at cosine ≥ 0.95**, plus
**125 within-split pairs**.

These are not look-alike radiographs. **133 of them sit at cosine 1.0000**, and the twelve
strongest were checked at the file level: **all twelve are byte-identical** — same MD5,
same pixels, mean absolute difference 0.00. TBX11K stores the same image under two
different filenames, in two different splits.

| Pair | Count | Why it matters |
|---|---|---|
| train → test | 84 | the challenge leaderboard partly scores models on their own training images |
| **train → val** | **52** | **leaks directly into our calibration and internal test, because both are carved from official val** |
| val → test | 22 | affects comparison between our internal test and the leaderboard |
| within train | 102 | inflates the effective training size; not a leak |
| within val / test | 21 | shrinks the effective evaluation n; belongs in the limitations section |

By class, the duplicates are overwhelmingly in **sick (non-TB)**: 94 of the cross-split
pairs are sick↔test and 45 are sick↔sick. One pair is tb↔tb, which matters out of
proportion to its count — there are only 800 TB images, so a duplicated TB case across
train and val inflates the positive class on the split everything is selected on.

## Why this is not an artefact of our method

Three checks, in increasing strength:

1. All 12,278 image IDs are unique, so no two files were collapsed by the ID scheme.
2. The embedding is deterministic: a re-run on a 64-image subsample reproduced the saved
   vectors at worst cosine 1.000000.
3. The pairs were verified at the byte level, not the embedding level. Cosine 1.0000 only
   pointed at them; `md5` and a pixel comparison confirmed them.

## What Person 1 must do before the split freeze

1. **Review all 158 candidates (P1-07)** and record a verdict per pair. The precedence
   rule in D06 is that the earlier split keeps the image and the later one loses it:
   train < development < calibration < internal_test < challenge_test.
2. **Removing the val-side copy of the 52 train→val pairs is not optional.** Leaving them
   means reporting a sensitivity measured partly on training images.
3. **The train→test pairs cannot be fixed** — we do not control the challenge test set.
   They are reported as a caveat on the leaderboard number, in the same breath as the
   Section 4.6 category-mapping caveat.
4. Re-run this search against the real C01 once it exists. τ does not move: D06 fixed it
   at 0.95 by a rule written before the sweep. Only the count changes, because
   train/development/calibration/internal_test is a finer partition than train/val.

## A second decision this forces: which list files?

TBX11K ships two sets of lists and they disagree:

| | train | val | test |
|---|---|---|---|
| `TBX11K_*.txt` | 6,600 | 1,800 | — |
| `all_*.txt` | 6,888 | 2,088 | 3,302 |

The difference is the 576 images in `imgs/extra/`, split 288/288. Our methodology quotes
6,600 / 1,800 / 2,800, so it assumed the `TBX11K_*` lists. This is a D01/D02 call for
Person 1 and it changes every split size, denominator and confidence interval in the
report, so it must be made before the freeze rather than discovered in Week 9.

**It does not rescue us from the duplicates.** Only 5 of the 158 pairs involve an
`extra/` image, and **46 of the 48 train→val pairs at τ=0.98 are between two images that
are in the core `TBX11K_*` lists**. Choosing the smaller lists removes the ambiguity
about dataset size; it does not remove the leak.

## Why this matters beyond our project

TBX11K is a CVPR 2020 benchmark and the standard public dataset for TB localisation. Any
published result that trained on its train split and evaluated on its val split has been
scored partly on images it was trained on. We have not seen this documented, and we are
not claiming priority without checking the literature properly — but it is worth a
paragraph in the paper either way, and it is exactly the kind of finding that makes a
methods section credible: we looked, and we say what we found.

## Reproducing

```bash
py -3 scripts/p2_03_embed.py --raw-dir data/raw/tbx11k/TBX11K/imgs \
    --variant full_clahe0 --out artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz --verify
py -3 scripts/p2_04_dedup.py --emb artifacts/p2/emb/radino_tbx11k_full_clahe0_v1.npz \
    --splits artifacts/p2/dedup/provisional_official_splits.csv \
    --out artifacts/p2/dedup/provisional_near_dup_official_v0.csv --official-splits --sweep
```

Candidate list: `artifacts/p2/dedup/provisional_near_dup_official_v0.csv`.
Marked provisional because it uses TBX11K's own train/val/test rather than C01, which
does not exist yet. It is **not** C17; C17 follows once Person 1 publishes the splits.

## One caveat on the embedding run

`preproc_version` on the C07 file reads `v0-p2-standin`, because Person 1's C04 image
store does not exist yet and Person 2 resized the raw images instead. The duplicate
finding is unaffected — byte-identical files are byte-identical under any preprocessing —
but the embeddings themselves must be regenerated from C04 before any score is reported
from them.


---

# Second finding: Shenzhen and Montgomery are already in TBX11K

Found 2026-10-06, after the external sets finished downloading. This one changes a
decision rather than adding a caveat.

Embedding all 800 NLM images and searching them against the 12,278 TBX11K images returns
**800 matches — every single external image has a twin inside TBX11K**, at cosine 0.9994
to 0.9996. Not 1.0000, because TBX11K redistributes them resampled to 512×512 while the
NLM originals are roughly 3000×3000; the same radiograph, processed differently.

It is not concealed. TBX11K has a folder called `imgs/extra/mc+shenzhen/`.

| Where the twin lives in TBX11K | Count |
|---|---|
| `imgs/test/` (renamed `unknown_NNNN`) | 400 |
| `imgs/extra/mc+shenzhen/train/` | 200 |
| `imgs/extra/mc+shenzhen/val/` | 200 |

## This settles the list-file decision

| Lists Person 1 chooses | External images we would train or select on |
|---|---|
| `TBX11K_*.txt` (6,600 / 1,800) | **0** |
| `all_*.txt` (6,888 / 2,088) | **200 in train, 200 in val** |

**Person 1 must use the core `TBX11K_*.txt` lists.** With the `all_*` lists we would train
on 200 Montgomery and Shenzhen images and then report "external validation" on those same
images. The external result would be meaningless and so would the recalibration study
built on it, which is the project's main claim.

This is no longer a question of which denominator to print in the report. It decides
whether the headline result is valid.

## What still needs saying in the paper

1. **400 of our 800 external images are in TBX11K's challenge test set.** We never train
   on test, so this is not a leak — but our external number and our leaderboard number
   are not independent measurements, and we must say so rather than present them as two
   corroborating results.
2. **Shenzhen and Montgomery are weaker as "external" sites than they appear.** The
   images were available to TBX11K's authors and 400 are inside its test split. They are
   still a genuine change of site, scanner and population, and with the core lists we
   never train on them, so the evaluation stands. But a reviewer will ask, and the honest
   answer must already be in the text.
3. **This strengthens the case for a third, genuinely independent site.** TB Portals has
   no relationship to TBX11K. The application should go out now rather than after the
   first draft.

## Reconciliation of the image counts, for Person 1's D01

```
3,800 health + 3,800 sick + 800 tb + 576 extra + 3,302 test = 12,278 files
extra = 472 PNG + 104 JPG
      = mc+shenzhen 200 train + 200 val   (all from NLM)
      + da+db        41 train +  31 val   (PNG)  + 104 JPG
all_train (6,888) = TBX11K_train (6,600) + 288 extra
all_val   (2,088) = TBX11K_val   (1,800) + 288 extra
```
