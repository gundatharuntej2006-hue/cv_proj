# Change-request draft: original image in the per-case PDF (R3 `make_pdf`, C14)

*Draft by P3, for P4 (contract steward). Not filed in `contracts/ccr/` by P3. If contracts-v1 is already tagged,
P4 files it as `contracts/ccr/CCR-NNN.md` (needs 4 approvals). Otherwise it can be settled in the freeze PR.*

## Problem

The P3-15 specification asks for the original image next to the Grad-CAM overlay in the per-case PDF. The frozen
R3 signature is `make_pdf(case_result) -> pdf_path`, and the C14 CaseResult has no field that points to the
original (or standardised 512 × 512) image. It only has `explainability.heatmap_path` and `overlay_path`.
`make_pdf` therefore cannot show the original image without an interface change.

What P3 does now, with no contract change:
- `make_pdf(case_result)` (frozen) produces an **overlay-only** report. The overlay already shows the X-ray, with
  the attention map semi-transparent on top.
- `make_pdf_with_original(case_result, *, image, out_path)` is a separate, non-frozen function that adds the
  image when the caller has it in memory (e.g. inside `run_case`).

## Options

1. **Add an image path to C14** (for example `explainability.image512_path`, pointing to a de-identified
   standardised PNG written by the pipeline under `runs/{run_id}/{case_id}_img512.png`).
   - Pros: the frozen `make_pdf(case_result)` can show image and overlay side by side, as P3-15 asks.
   - Cons: a C14 schema change (bump `schema_version`), P4's `run_case` must write the file, and P1 and P4
     consumers must accept the new field. The path must never be the raw input path (PHI risk); it must be a
     pipeline-written file named by case_id.
2. **Accept overlay-only PDFs.**
   - Pros: no contract change. The X-ray is still visible under the overlay. P4 may call
     `make_pdf_with_original` directly when it wants both panels.
   - Cons: the plain image is not shown separately, which slightly reduces readability for the reviewer.

## Affected consumers

P4 (C14 schema, `run_case`), P1 (dashboard reads C14), P3 (`make_pdf`).

## P3 recommendation

Option 1 if P4 already writes a standardised image per case; otherwise option 2 for this release.
