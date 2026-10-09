# P3: what is blocked on other people (and on me)

Status on 2026-10-09, after the audit fixes. Check IDs refer to docs/p3_verification_report.md.

## Blockers by owner

| Item | Owner | Unblocks checks | Exact thing to ask | Status |
|---|---|---|---|---|
| Team git repository (`main`) | P1 | A1, A5, B4, H3 (on the team repo) | Create the team repo (P1-01) and share the URL so P3 can push `p3/audit-fixes` | waiting |
| Register D-FINE as a submodule | P1 | A5 | Add `third_party/D-FINE` as a submodule pinned at `956d1709314c2c6a4df6f34de232054578a7449f` (`.gitmodules` is a root file), or approve P3 doing it | waiting |
| Root `.gitignore` entries | P1 | A4 (hygiene on the team repo) | Ignore `artifacts/p3/det/checkpoints/`, `artifacts/p3/_preview_synthetic/`, `mock/artifacts/`, `mock/logs/`, `logs/`, `runs/` (P3 already has nested `.gitignore` files inside its own folders) | waiting |
| `requirements/base.txt`, `pyproject.toml` with a `gpu` marker | P1 | A2/A3 in CI | Publish base requirements (CI now installs CPU torch itself) and register the `gpu` pytest marker | waiting |
| C01/C04/C05/C06 data + uncropped box file | P1 | B2, E3, E5, C1, C2, C3, D4, F2, F3, F5, J2 | Release the split file, the processed store, the lung ROI table, C06, and `coco_{split}_full.json` with all original boxes (contract notes item 11) | waiting |
| Dashboard | P1 | K3 | Tell P3 when the dashboard can open an overlay PNG and a PDF at 1366×768 | waiting |
| `classifier_module()` with `preprocess()` and `tb_index` | P2 | D5, F2, F3 | R2 must expose the same crop preparation `classify()` uses, plus the TB output index (contract notes item 13) | waiting |
| Full-image comparator via R2 | P2 | F5 | `classifier_module(model_id=None)` so P3 can load the full-image model for methodology §14 (contract notes item 14) | waiting |
| D08 (CLAHE) and D09 (margin) locked | P2 | B2, E3 (locked-preprocessing runs only) | Lock D08/D09 so the D16 runs can be repeated on locked preprocessing | waiting |
| D14 model registry and its layout | P2 | F5, H-series later | Publish `decisions/p2_model_registry.json` and document where model ids and SHA-256 live | waiting |
| tbcore: validators, disclaimer texts, guard, seed | P4 | D2c, A2 (hard step), G2, H1 | Publish tbcore; confirm or correct the names in `src/p3_ops/_tbcore_adapter.py` (`seed.set_all`, `guard.require_unlocked`, `validators.validate`, `disclaimer.DISCLAIMER` / `SCOPE_HONESTY`) | waiting |
| C16 decision schema | P4 | B2 | Publish `contracts/decision.schema.json` | waiting |
| 7 CaseResult fixtures (P4-02) | P4 | G2, C4 | Publish the 7 fixtures; P3's tests currently use stand-ins | waiting |
| C12 triage table (development) | P4 | J2 | Publish `artifacts/p4/triage/triage_tbx11k_development_v1.parquet` (or its mock) | waiting |
| R4 `run_batch` / `run_case` | P4 | I1, I2, K1, K2, C4 | Publish R4 so P3 can run E6 and the integration checks | waiting |
| Gate and tokens (P4-09) | P4 | H1–H4 | Publish the gate API: how a token is checked and consumed, and the audit-line format | waiting |
| Detector on/off flag source | P4 | D6 | Decide where `detector_enabled` lives (D21 key, C14 field or a new P4 decision) | waiting |
| CCR: original image in the PDF | P4 | (FIX 3 follow-up) | Decide option 1 or 2 in `docs/p3_ccr_drafts/make_pdf_original_image.md` | waiting |
| `run_id` in C14 | P4 | (logger) | Add `run_id` to CaseResult (contract notes item 1) | waiting |
| T4 feasibility run (P3-01) | me | J1, D2b | Run `notebooks/p3/p3_01_dfine_feasibility.ipynb` on a Kaggle T4; fill `epochs` / `batch_size` | to do |
| D-FINE training, both sizes | me | B2, B6, E3, E5 | After P1 data and D08/D09: train `dfine_512.yaml` and `dfine_640.yaml` | to do |
| Deliberate-kill test | me | I3 | `docs/p3_kaggle_runbook.md` section 10 | to do |
| GPU tests on the RTX 4050 | me | F6 | `pytest tests/p3_xai/test_gradcam.py -m gpu -s` and `make p3-latency-detect` | to do |
| D19 sanity rule: signed vs absolute Spearman | me | B4 (D19) | Decide whether to change "mean Spearman < 0.5" to "mean abs Spearman < 0.5" BEFORE committing D19's criterion | decision needed |
| D20 confirmation | me | B2 | Confirm the licences after the T4 numbers; then lock D20 | to do |
| Guide sessions | me + guide | B5, J2 | D18 acknowledgement; Grad-CAM review session | to do |

## Messages to paste

**To P1:**
> Hi! P3 here. Could you (1) create the team repo so I can push my branch `p3/audit-fixes`, (2) register D-FINE as a
> submodule at `third_party/D-FINE`, pinned at `956d1709314c2c6a4df6f34de232054578a7449f` (or tell me to do it, since
> it adds `.gitmodules` at the root), (3) add these to the root `.gitignore`: `artifacts/p3/det/checkpoints/`,
> `artifacts/p3/_preview_synthetic/`, `mock/artifacts/`, `mock/logs/`, `logs/`, `runs/`, (4) publish
> `requirements/base.txt` and register a `gpu` marker in `pyproject.toml`, and (5) tell me when C01/C04/C05/C06 are
> ready, ideally with an uncropped box file `coco_{split}_full.json` containing all original boxes (my contract notes
> item 11). Thanks!

**To P2:**
> Hi! P3 here. For Grad-CAM I need `classifier_module()` to expose `preprocess(crop_uint8) -> tensor[1,3,H,W]` (the same
> preparation `classify()` uses) and `tb_index`, plus a way to load the full-image comparator, e.g.
> `classifier_module(model_id=...)`, for the methodology §14 "does lung cropping help?" row. Could you also tell me
> when D08/D09 are locked (I must re-run the detector on the locked settings) and what layout D14 will have? Details
> are in `docs/p3_contract_notes.md` items 13 and 14. Thanks!

**To P4:**
> Hi! P3 here. My audit is waiting on tbcore (validators, the disclaimer and scope texts, the sealed-split guard,
> seed), `contracts/decision.schema.json`, the 7 CaseResult fixtures (P4-02), the development C12, R4
> `run_batch`/`run_case`, and the gate API (P4-09). All my tbcore calls go through one adapter
> (`src/p3_ops/_tbcore_adapter.py`), so please confirm or correct the names in `ASSUMED_API`. Two decisions needed:
> (1) where the detector on/off flag lives (D21 key, C14 field or a new decision) and (2) the original image in the
> per-case PDF: `docs/p3_ccr_drafts/make_pdf_original_image.md` has two options. Could you also add `run_id` to C14?
> Thanks!

## Local git history (FIX 7) and moving it to the team repo

There is no team repo yet, so P3 keeps a local repository in this working copy:

| Commit | Date | What |
|---|---|---|
| `2e91fe6` | 2026-10-09 12:39 | `main`: baseline plan documents as received |
| `c98b234` | 2026-10-09 12:39 | P3-06: pre-register D16 criterion |
| `50be8b7` | 2026-10-09 12:39 | P3-07: pre-register D17 criterion |
| `61ef02a` | 2026-10-09 12:39 | P3-07: pre-register D18 rule |
| `11b45b5` | 2026-10-09 12:39 | P3 work P3-00..P3-18 (snapshot) |
| `894721b` … | 2026-10-09 | `P3-fix …` commits, one per audit fix |

D19's criterion is **not** committed yet: it waits for the signed vs absolute Spearman decision above.

**Moving these commits onto the team repo** (keeps every commit, hash-independent order and dates; no squash, no rewrite):

```
git remote add team <TEAM_REPO_URL>
git fetch team
git rebase --onto team/main main p3/audit-fixes   # replays my commits, in order, on top of the team's main
git push team p3/audit-fixes                      # then open a PR from p3/audit-fixes
```

- Rebase gives the commits new hashes but keeps their order, messages and author dates, so "criterion committed
  before result" stays provable (`python -m p3_det.decisions verify ...`). Record the old → new hash mapping from
  `git log` in the PR description.
- If the team's `main` already contains the three plan documents, the local `2e91fe6` baseline is simply not
  replayed (`--onto` starts after it).
- Do not squash-merge the criterion commits into the result commits. Ask P4 to use a merge or rebase-merge for this
  PR, so the pre-registration commits stay visible in the team history.
