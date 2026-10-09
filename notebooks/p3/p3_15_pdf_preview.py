"""P3-15: build the 7 example PDFs and render page 1 of each to PNG for a layout check.

    python notebooks/p3/p3_15_pdf_preview.py

Uses the test-only stand-ins (7 CaseResult scenarios, tbcore disclaimer placeholders) and a synthetic
image, so the PDFs show PLACEHOLDER disclaimer/scope wording until P4's tbcore exists.
Writes artifacts/p3/_preview_synthetic/ops_pdf_preview/SYNTHETIC_{scenario}.pdf/.png and SYNTHETIC_contact_sheet.png.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "tests" / "p3_mock"):
    sys.path.insert(0, str(p))

import tbcore_standin  # noqa: E402
from p3_case_results import P4_SCENARIOS  # noqa: E402

OUT = REPO / "artifacts" / "p3" / "_preview_synthetic" / "ops_pdf_preview"


class _Patch:
    def setitem(self, d, k, v):
        d[k] = v


def main() -> int:
    import pypdfium2 as pdfium

    from p3_det.data.fake_p1 import _image
    from p3_ops import api
    from p3_xai import api as xai_api
    tbcore_standin.install(_Patch())
    case_root = Path(tempfile.mkdtemp(prefix="p3_15_case_"))
    os.environ["TB_CASE_ROOT"] = str(case_root)
    r = np.random.default_rng(3)
    img = _image(r, (60, 40, 450, 470), [[300.0, 88.0, 372.0, 160.0]])
    yy, xx = np.mgrid[0:512, 0:512]
    heat = np.exp(-((xx - 336) ** 2 + (yy - 124) ** 2) / (2 * 38 ** 2))
    heat[:40, :] = 0
    for cr in P4_SCENARIOS.values():
        if cr.get("explainability"):
            xai_api.render_overlay(img, heat.astype(np.float16), (cr.get("detector") or {}).get("boxes"),
                                   out_path=case_root / cr["explainability"]["overlay_path"], case_id=cr["case_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for kind, cr in P4_SCENARIOS.items():
        pdf = Path(api.make_pdf(cr, image=img, out_path=OUT / f"SYNTHETIC_{kind}.pdf"))
        page = pdfium.PdfDocument(str(pdf))[0]
        png = page.render(scale=1.6).to_pil()
        png.save(OUT / f"SYNTHETIC_{kind}.png")
        thumbs.append(png)
        print(f"{kind:22s} {pdf.stat().st_size / 1024:6.0f} KB  -> {pdf}")
    w, h = thumbs[0].size
    cols = 4
    sheet = Image.new("RGB", (cols * w // 2, 2 * h // 2), "white")
    for k, t in enumerate(thumbs):
        sheet.paste(t.resize((w // 2, h // 2), Image.Resampling.LANCZOS), ((k % cols) * w // 2, (k // cols) * h // 2))
    sheet.save(OUT / "SYNTHETIC_contact_sheet.png")
    print(f"contact sheet -> {OUT / 'SYNTHETIC_contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
