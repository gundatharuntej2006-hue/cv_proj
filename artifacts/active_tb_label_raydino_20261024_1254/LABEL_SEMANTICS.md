# Label semantics — is `active_tb_label` the documented target for active TB?

**Verdict: YES.** The decision gate in Phase 1 passes, on primary-source evidence, not on
inference from a score.

---

## 1. The primary source

TBX11K is distributed with COCO-format annotations. The `categories` block of
`data/raw/tbx11k/TBX11K/annotations/json/TBX11K_train.json` and `..._val.json` is the
dataset authors' own declaration of what a lesion box means:

| id | name | supercategory |
|---|---|---|
| 1 | **ActiveTuberculosis** | Tuberculosis |
| 2 | **ObsoletePulmonaryTuberculosis** | Tuberculosis |
| 3 | PulmonaryTuberculosis | Tuberculosis |

`info`: `{"description": "TBX11K Dataset", "version": "1.0", "year": 2020,
"contributor": "Yun Liu, Yu-Huan Wu, Yunfeng Ban, Huifang Wang, Ming-Ming Cheng",
"url": "http://mmcheng.net/tb"}`.

"Obsolete pulmonary tuberculosis" is the radiological term for **healed / inactive**
disease. The dataset therefore distinguishes active from inactive TB *at the annotation
level*, and that distinction is the dataset's, not this project's.

The three-class folder structure (`health/`, `sick/`, `tb/`) is a separate axis and does
**not** encode activity: every one of the 800 `tb/` images is in the same folder whether
its boxes are category 1 or category 2.

## 2. The derivation, stated as a rule

Both targets are deterministic functions of the lesion-box category set:

```
box set {1}      -> category active_tb         -> tb_label 1, active_tb_label 1
box set {1, 2}   -> category active_latent_tb  -> tb_label 1, active_tb_label 1
box set {2}      -> category latent_tb         -> tb_label 1, active_tb_label 0
folder health/   -> category healthy           -> tb_label 0, active_tb_label 0
folder sick/     -> category sick_non_tb       -> tb_label 0, active_tb_label 0
```

`tb_label  = category in {active_tb, latent_tb, active_latent_tb}`
`active_tb_label = category in {active_tb, active_latent_tb}`

Verified over all 8,400 trainval rows: the mapping is deterministic and both stored
columns are reproducible from `category` with **one** exception, `tb1199`
(`tables/phase01_audit.json`; see PHASE0 §4).

## 3. Crosswalk

Counted from the annotation files directly (not from a derived CSV):

| box category set | images | meaning |
|---|---|---|
| `[1]` | 630 | active only |
| `[1, 2]` | 30 | active **and** obsolete lesions on the same radiograph |
| `[2]` | **139** | obsolete only — **the mismatch set** (140 under C01, which also puts `tb1199` here) |
| none | 1 | `tb1199` |
| `[3]` | **0** | never occurs in trainval |

Full split-by-category-by-label crosswalk: `tables/target_crosswalk.csv`.

| `tb_label` | `active_tb_label` | n | which images |
|---|---|---|---|
| 0 | 0 | 7,600 | 3,800 healthy + 3,800 sick_non_tb |
| 1 | 1 | 661 | active + mixed |
| **1** | **0** | **139** | obsolete-only — positive under one target, negative under the other. **140 under the C01 split file**, which also assigns `tb1199` to `latent_tb`; the experiment uses C01, so 140 images are relabelled |
| 0 | 1 | 0 | — (impossible by construction) |

## 4. Why this matters for the system as deployed

The Vietnam reference standard is **Xpert MTB/RIF Ultra**, a microbiological test for
*active* infection. A participant with healed TB and no current disease is Xpert
**negative**.

Training on `tb_label` therefore teaches the head to call the radiographic signature of
**inactive** disease positive, and the head is then evaluated against a standard that
calls exactly those participants negative. That is a target-definition mismatch between
the training task and the evaluation task, and it is the mechanism the CLEAR concept audit
independently pointed at: among TB-negative Vietnamese participants, scarring similarity
correlates with the classifier score at rho = 0.566, while scarring similarity itself
carries **no** information about Xpert status (AUROC 0.494).

Two findings pointing the same way is a reason to test the change. It is **not** proof
that the change will help, and the concept correlations do **not** establish that the
classifier causally uses a scarring cue.

## 5. Decision gate

| requirement | met | evidence |
|---|---|---|
| the dataset's own documentation supports an active/inactive distinction | **yes** | COCO `categories`, ids 1 and 2 |
| the mapping from source rows to the target is deterministic | **yes** | 8,400/8,400 rows, 1 documented exception |
| every target class is accounted for | **yes** | 630 + 30 + 139 + 1 = 800 TB; + 3,800 + 3,800 = 8,400 |
| no ambiguous class is silently assigned | **yes** | category 3 has 0 images; `tb1199` is documented, not hidden |
| the mapping was chosen before any score was seen | **yes** | fixed in the protocol, hashed before fitting |

**Phase 2 may proceed.**

## 6. What this does not settle

- Whether `active_tb_label` yields a **better** model. That is the experiment, and a
  higher AUROC would not retrospectively prove the mapping correct.
- Whether TBX11K's own active/obsolete reading is accurate. It is a radiologist reading
  of a radiograph, not a microbiological result, and this project cannot audit it.
- Whether the 30 mixed `[1, 2]` images should count as active. They are treated as
  positive here because an active lesion is present; that is a judgement, declared in
  advance, and it moves 30 of 8,400 images.
