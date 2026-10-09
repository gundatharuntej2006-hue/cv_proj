# SYNTHETIC previews: never cite, never ship

Everything in this folder is made from synthetic images, fake P1 data, test-only stand-ins for P4's CaseResult
fixtures and a PLACEHOLDER disclaimer (not the tbcore text). It exists only to check layouts and plumbing by eye.

- Never cite any file here in a report, decision or table.
- Never give any file here to a user, a clinician or the dashboard.
- The files are not committed (see .gitignore); regenerate them with:
  - `python notebooks/p3/p3_04_viz_check.py --fake`  -> det_viz_check/
  - `python notebooks/p3/p3_11_overlay_preview.py`   -> xai_overlay_preview/
  - `python notebooks/p3/p3_15_pdf_preview.py`       -> ops_pdf_preview/
