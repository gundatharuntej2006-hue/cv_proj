# P3: messages to send (prepared 2026-10-10, not sent)

Paste-ready chat messages built from `docs/p3_blockers.md`. Each is under 130 words.

## To P1

> Hi P1, P3 here (2026-10-10). Four asks:
> 1. Could you create the team repo and share the URL? I'll push my branch `p3/audit-fixes`.
> 2. Once it exists, I will register D-FINE as a submodule at `third_party/D-FINE`, pinned at `956d170`. That adds
>    `.gitmodules` at the root, so please confirm you're OK with that.
> 3. Please add `artifacts/p3/det/checkpoints/` to the root `.gitignore` (checkpoints must never be committed).
> 4. When will the dashboard be able to open an overlay PNG and a per-case PDF? I need it for my 1366×768 display
>    check (K3).
>
> Thanks!

## To P2

> Hi P2, P3 here (2026-10-10). Three asks for Grad-CAM:
> 1. Could `classifier_module()` expose `preprocess(crop_uint8) -> tensor[1,3,H,W]` (exactly what `classify()` uses)
>    and `tb_index` (which output is TB)?
> 2. How do I load the full-image comparator by `model_id`, e.g. `classifier_module(model_id=...)`? I need it for the
>    "does lung cropping help?" row (methodology §14).
> 3. What dates do you expect for locking D08 (CLAHE) and D09 (margin), and for publishing D14 (model registry)? I
>    have to re-run the detector on the locked preprocessing.
>
> Details: `docs/p3_contract_notes.md` items 13–14. Thanks!

## To P4

> Hi P4, P3 here (2026-10-10). Asks, all blocking checks:
> 1. Publish tbcore and the disclaimer text; please confirm the names in `src/p3_ops/_tbcore_adapter.py`.
> 2. Publish the C16 schema (`contracts/decision.schema.json`).
> 3. Publish the 7 CaseResult fixtures.
> 4. Publish the development C12 triage table.
> 5. Publish `run_batch` / `run_case` (R4).
> 6. Publish the gate and token API.
> 7. Where does the detector on/off flag live?
> 8. Please decide on the CCR draft `docs/p3_ccr_drafts/make_pdf_original_image.md`.
> 9. Please allow a rebase-merge or a merge commit (not squash) for my PR. Squash would erase my pre-registration
>    commits. Tags `p3-prereg-D16` to `p3-prereg-D19` also preserve them.
>
> Thanks!
