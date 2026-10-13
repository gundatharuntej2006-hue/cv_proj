# Which public TB datasets could serve as a genuinely independent site?

**Owner:** Person 2 · **Date:** 2026-10-07 · **Feeds:** D04 (external source), P4-14, P4-17, the paper's related work

## Why this question needs asking at all

The project's headline result is that a triage threshold set at one site does not transfer,
and that a small local sample re-establishes it. That claim is only testable against a site
that is genuinely independent of TBX11K.

We learned the hard way that "external" is not the same as "independent". Shenzhen and
Montgomery were listed as our external validation sets throughout the methodology. They are
**inside TBX11K** — all 800 of them, 400 in `imgs/extra/mc+shenzhen/` and 400 in the test
split (see `p2_duplicate_finding.md`). Every candidate below is therefore judged on one
criterion before any other: is it already in TBX11K?

## The landscape

| Dataset | TB cases | Total | Access | Independent of TBX11K? |
|---|---|---|---|---|
| Shenzhen + Montgomery | 394 | 800 | open | **No** — verified inside TBX11K at cosine 0.999 |
| DA and DB | 125 | 278 | Kaggle / SourceForge | **No** — TBX11K ships `imgs/extra/da+db/` |
| TB Chest X-ray Database (Qatar/Dhaka) | 700 | 4,200 | Kaggle | **No** — assembled from Shenzhen, Montgomery, Belarus and NIAID, so it inherits the overlap |
| Belarus | 305 | 306 (169 patients) | see below | Probably — different country, programme and equipment |
| VinDr-CXR | 479 | 18,000 | PhysioNet, credentialed | Probably — Vietnam, independent collection |
| VinDr-PCXR | 15 | 9,125 | PhysioNet, credentialed | Paediatric, too few TB cases to be useful |
| NIAID TB Portals | ~3,500 | 28,181 images, 40 sites, 16 countries | data usage agreement | Yes |
| PadChest | 176 studies | 160,861 | registration | Probably, but 1.02 TB for 176 TB cases |

**The public TB imaging landscape is small, and most of it is the same images recirculated.**
Three of the eight entries are wholly or partly inside TBX11K. That is worth a sentence in the
paper: cross-dataset validation in this field is frequently not cross-dataset at all.

## The two real options

**TB Portals (NIAID)** is the right answer. 28,181 images across 40 sites in 16 countries, a
proper data usage agreement, and — importantly — **Belarus is part of it**. One application
obtains the most accessible independent set *and* the multi-country breadth that makes a
threshold-transfer study convincing. Imaging requests go to `TBPortalsInfo@niaid.nih.gov`;
the letter is drafted in `Downloads/TB_data_access_pack.md`.

**VinDr-CXR** is the backup: 479 TB cases among 18,000 radiographs from Vietnam. PhysioNet
credentialing requires CITI human-subjects training, which takes weeks, so it should start
now if it is going to be used at all.

## On the Belarus GitHub copy

The images survive as `belarus.zip` (48 MB) in `frapa/tbcnn`, a research repository carrying
an MIT licence. **That MIT licence covers the repository's code, not the radiographs.** The
authors were not in a position to relicense medical images collected by the Belarus Ministry
of Health under a NIAID drug-resistance study, and the original portal is no longer
reachable. Taking the convenient copy would repeat exactly the mistake we refused to make
with Open-i.

The licensed route to the same images is TB Portals.

## A method worth keeping

Any candidate external set must be checked for overlap against TBX11K **before** it is used,
not after. The procedure already exists and is what caught Shenzhen:

1. Embed the candidate with `scripts/p2_03_embed.py` (full-image variant).
2. Run `p2_gate.dedup.find_near_duplicates` against the TBX11K embeddings with a custom
   split order, as in the cross-dataset check.
3. Anything at cosine ≥ 0.95 is the same radiograph. Shenzhen matched at 0.9994–0.9996
   because TBX11K redistributes it resampled.

Roughly ten minutes per candidate, and it is the difference between an external validation
and a very expensive way of testing on your training data.

## What this means for the schedule

Only one action moves this forward, and it is not a technical one: **send the TB Portals
application.** The public landscape has been mapped and contains nothing else both
independent and reachable within the project's three months. Further dataset hunting has
hit diminishing returns.

Until it arrives, the external-validation sections (P4-14, P4-17) have no valid site, and
the honest fallback is to report the Shenzhen and Montgomery results with the overlap stated
plainly as a limitation rather than presented as external validation.
