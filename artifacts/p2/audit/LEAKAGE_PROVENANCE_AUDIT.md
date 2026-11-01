# Leakage and provenance audit

**Owner:** Person 2 · **Date:** 2026-10-21 · **Scope:** read-only. No code changed, no runs
re-executed, no artifacts overwritten. New files are confined to `artifacts/p2/audit/`.

Machine-readable companion: `provenance_table.csv` (14 decisions).
Case-level gate record: `vietnam_case_level_gate_audit.csv` (2,708 rows).

---

## 1. Method

Provenance was established from three independent sources and only claimed where they agree:

1. **Commit timestamps** — `git log --format="%h|%ad|%s"`.
2. **First-appearance of each script** — `git log --diff-filter=A`.
3. **Artifact modification times** — when a result actually came into existence.

A decision is called **precommitted** only where the selection rule was committed to git
**before** the result that would have informed it existed. The existence of code in git is
*not* treated as evidence of precommitment, as instructed.

---

## 2. The critical timing evidence

| event | timestamp | source |
|---|---|---|
| TUNE/REPORT split created | **2026-10-20 18:51** | artifact mtime |
| split + sweep script + **scoring script** committed (`1c81573`) | **2026-10-20 19:00** | commit time |
| sweep results first exist (`mil_sweep.json`) | **2026-10-20 19:46** | artifact mtime |
| REPORT half read (`mil_final.json`) | **2026-10-20 19:47** | artifact mtime |

**The scoring script `p2_33_mil_final.py` was committed 46 minutes before any sweep result
existed, and 47 minutes before the REPORT half was read.** The reporting rule therefore
could not have been retrofitted to the winner.

**Caveat on strength of evidence.** Artifact mtimes record the *last* write. Earlier
overwritten sweep runs are not recoverable, so these timestamps are *consistent with*
precommitment and show no contrary evidence, rather than proving no earlier run occurred.
The commit at 19:00 is the stronger artifact because it is immutable and predates all
recorded execution.

---

## 3. Provenance table (summary — full version in the CSV)

| decision | value | saw full Vietnam? | precommitted? | status |
|---|---|---|---|---|
| D08 CLAHE | off | no | n/a (pre-dates Vietnam) | verified |
| D09 crop (original) | full image | no | n/a | verified |
| **D09 revisited → lung crop** | lungs-only | no (used Shenzhen/Montgomery) | **no — post-hoc** | verified |
| D13 OOD threshold | 0.70 | no | **no** — fitted to two alien pools | verified |
| D11 head type | binary | no | criterion pre-registered, amended and dated | verified |
| **Negative composition** | healthy-only | **YES** | **no** | verified |
| **Focal pooling (decision to use regions)** | yes | **YES** | **no** | verified |
| TUNE/REPORT split | 1,352 / 1,356 | labels + scanner only, no model scores | **yes** | verified |
| **Pooling grid + aggregator** | 12×12, max | no — TUNE only | **YES** | verified |
| Not switching to 18×18 + lse | kept 12×12 max | no | **yes** — honoured despite higher value | verified |
| REPORT read | 0.7242 | — | n/a (the reported result) | verified |
| Attention-MIL | rejected | no | no — run *after* the REPORT read, labelled exploratory in-code | verified |
| Gate operating threshold (audit) | 2.4766 | no — TUNE only | no — refitted for the audit | verified |
| TBX11K site metadata | **does not exist** | n/a | n/a | verified |

---

## 4. The contamination boundary, stated precisely

**Two decisions were informed by the full Vietnam cohort, including patients later assigned
to REPORT:**

1. **Negative composition → healthy-only** (2026-10-20 09:22). `p2_29_hard_negatives.py`
   scored all 2,708 participants; the TUNE/REPORT split did not exist until 18:51 that day.
2. **The decision to use regional pooling at all** (2026-10-20 11:14). The preliminary 6×6
   run in `p2_focal_pooling.md` was scored on the full cohort.

**One decision was precommitted:** the *choice of grid and aggregator* within focal pooling,
selected on TUNE alone under a rule committed beforehand.

### What this means for the 0.7242 figure

- It **is** a clean estimate of *"given this configuration, does 12×12 max beat the
  alternatives, and does that choice generalise within the cohort?"* Selection optimism was
  **−0.0005** (TUNE 0.7237, REPORT 0.7242), which is strong evidence the pooling choice is
  not overfitted.
- It is **not** a pristine untouched estimate of the configuration's absolute generalisation,
  because the two decisions above saw REPORT patients.
- **No retroactive repair is possible.** A split cannot be made untouched after the fact.
  An independent confirmation requires a cohort that has never been scored — none currently
  exists in this project.

### Defensible claims, in descending order of strength

1. **Focal pooling beats global pooling** — internal `mean` control flat at 0.636–0.642
   across all grids while `max` reaches 0.7239; dissociation is mechanistic, not a capacity
   artifact. *Strongest claim in the project.*
2. **The border shortcut does not transfer** — 0.9094 at home, 0.52–0.62 across four
   external sites. Verified pre-Vietnam, so untouched by this contamination.
3. **Benchmark performance does not predict screening performance** — 0.9949 → 0.6151 with a
   commercial comparator at 0.7862 on identical patients. Established *before* any tuning.
4. **0.7242 on the REPORT half** — defensible as an observed result for a configuration
   partly chosen on the full cohort. Must be reported with that caveat attached.

---

## 5. Gate workflow counts, reconciled

Population denominators are stated because an earlier version of this table omitted them and
a reader reasonably mis-derived the net workload.

| quantity | population | without gate | with gate |
|---|---|---|---|
| referrals | all 2,708 | **929** | **878** |
| — true positives | 280 TB | 174 | **167** |
| — false positives | 2,428 non-TB | 755 | 711 |
| deferred for review | all 2,708 | 0 | **284** |
| — TB deferred | 280 TB | 0 | **21** |
| — non-TB deferred | 2,428 non-TB | 0 | 263 |
| **total needing human time** | all 2,708 | **929** | **1,162** |

Net workload = (878 + 284) − 929 = **+233 cases**, computed on the all-case population
throughout. The alternative arithmetic 711 + 284 − 755 = 240 combines a non-TB-only count
with an all-case count and is not a meaningful quantity. **No seven-case discrepancy exists;
the 7 is a different number entirely** — see below.

### The 7 TB patients diverted from a correct referral

These are TB cases the classifier scored **above** the referral threshold — correct
detections — which the gate then intercepted:

| pid | scanner | half | classifier score (thr 2.4766) | OOD score (thr 0.70) |
|---|---|---|---|---|
| 3015671 | 1 | REPORT | 4.5653 | 0.7339 |
| 3453706 | 3 | TUNE | 4.4360 | 0.7127 |
| 5178054 | 1 | REPORT | 2.9328 | 0.7346 |
| 6142577 | 1 | REPORT | 3.4210 | 0.7117 |
| 6602320 | 1 | REPORT | 3.4506 | 0.7169 |
| 7143518 | 3 | TUNE | 2.5157 | 0.7114 |
| 8117332 | 1 | REPORT | 4.2889 | 0.7197 |

Every one sits **marginally** over the OOD threshold (0.711–0.735 against 0.70) while
scoring confidently positive. The remaining 14 of the 21 deferred TB cases would not have
been referred anyway.

**These are deferrals, not proven missed diagnoses.** Each enters a human review pathway
whose outcomes this project has no data for. Whether they are eventually diagnosed is
unknown and is not estimated anywhere.

### New finding: the gate is scanner-biased

| scanner | n | deferred | rate | TB n | TB deferred | TB rate |
|---|---|---|---|---|---|---|
| 1 (JPI Healthcare) | 408 | 82 | **20.10%** | 44 | 9 | **20.45%** |
| 2 (DRTECH) | 1,332 | 92 | 6.91% | 151 | 7 | 4.64% |
| 3 (Konica Minolta) | 968 | 110 | 11.36% | 85 | 5 | 5.88% |

**Scanner 1 is deferred at roughly three times the rate of scanner 2**, and 5 of the 7
diverted TB patients are on it despite it being only 15% of the cohort. The gate is
substantially detecting *which machine took the image*, not whether the image is assessable.
Since scanner does not predict TB status in this cohort (AUC 0.4722), this is a pure
equity cost: patients are deferred according to which clinic they attended.

This was not visible in any aggregate figure and is new to this audit.

---

## 6. TBX11K site metadata — confirmed absent

The COCO annotation files (`annotations/json/*.json`) contain `info`, `licenses`,
`categories`, `images`, `annotations`; the `images` records carry `id`, `file_name`,
`width`, `height`, `date_captured`, `license`, `coco_url`, `flickr_url` — **no site,
hospital, scanner or device field**. The list files carry paths only.

**Hospital-held-out validation on TBX11K is not constructible.** Any such split would be
invented. Scanner-held-out validation is possible only on the Vietnam cohort, which does
carry `rad_system`.

---

## 7. Unresolved questions

1. **No untouched external cohort exists.** Independent confirmation of 0.7242 remains
   outstanding and cannot be manufactured from existing data.
2. **Earlier sweep runs are unrecoverable.** Artifact mtimes show last writes only;
   precommitment is evidenced by the 19:00 commit rather than by proving no earlier run.
3. **Human-review outcomes are unavailable**, so the fate of the 21 deferred TB patients —
   including the 7 diverted from correct referral — cannot be determined.
4. **Whether focal pooling helps the curated sites is unknown.** Only Vietnam has been
   scored under the regional representation.
5. **The mechanism behind healthy-only negatives outperforming is undetermined** — label
   noise, suppression of transferable features, or a third acquisition axis remain open.
6. **Scanner-bias generality is unknown.** It is measured on one cohort with three devices.

---

## 8. Strongest supported claims

> Focal pooling over a 12×12 region grid substantially outperforms global average pooling for
> tuberculosis classification on a frozen RAD-DINO backbone, with an internal control
> establishing the gain is not a capacity artifact, and with near-zero selection optimism on
> a pre-committed held-aside half.

> A border-only shortcut reaching AUC 0.9094 on TBX11K collapses to 0.52–0.62 across four
> independent sites, so benchmark performance on this dataset is substantially
> non-pathological in origin.

> Benchmark AUC does not predict screening-cohort AUC: 0.9949 against 0.6151–0.7242, with a
> certified commercial system at 0.7862 on identical patients.

> A k-NN OOD gate tuned against non-radiograph inputs is net-harmful when placed in front of
> a screening population: +233 human reviews, 21 TB patients deferred, 7 of them diverted
> from a correct referral, and a threefold deferral disparity between scanners.

**Not supported:** any claim that this system is deployable, clinically validated, or that
0.7242 is an untouched estimate of generalisation.
