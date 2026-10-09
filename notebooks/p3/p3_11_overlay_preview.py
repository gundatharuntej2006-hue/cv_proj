"""P3-11: render an example overlay and a 1366x768 dashboard preview for a visual legibility check.

    python notebooks/p3/p3_11_overlay_preview.py

Synthetic X-ray, heatmap and boxes only (no real or patient data). Writes:
    artifacts/p3/_preview_synthetic/xai_overlay_preview/SYNTHETIC_overlay_full.png        native size (1024x1204)
    artifacts/p3/_preview_synthetic/xai_overlay_preview/SYNTHETIC_dashboard_1366x768.png  fitted into 683x768
Open SYNTHETIC_dashboard_1366x768.png at 100% zoom on a 1366x768 screen (or the browser at that window size).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_det.data.fake_p1 import _image  # noqa: E402
from p3_xai import api, overlay  # noqa: E402

OUT = REPO / "artifacts" / "p3" / "_preview_synthetic" / "xai_overlay_preview"


def main() -> int:
    r = np.random.default_rng(3)
    lesions = [[150.0, 140.0, 205.0, 190.0], [320.0, 260.0, 360.0, 330.0], [300.0, 380.0, 330.0, 405.0]]
    img = _image(r, (60, 40, 450, 470), lesions)
    yy, xx = np.mgrid[0:512, 0:512]
    heat = sum(w * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
               for w, cx, cy, s in ((1.0, 178, 165, 35), (0.7, 340, 295, 45), (0.35, 250, 420, 60)))
    crop = (60, 40, 450, 470)
    mask = np.zeros((512, 512), bool)
    mask[crop[1]:crop[3], crop[0]:crop[2]] = True
    heat = np.where(mask, heat, 0) / heat[mask].max()
    boxes = [{"xyxy": [148, 138, 207, 193], "score": 0.83}, {"xyxy": [318, 262, 362, 332], "score": 0.61},
             {"xyxy": [298, 378, 331, 407], "score": 0.44}]

    OUT.mkdir(parents=True, exist_ok=True)
    full = Path(api.render_overlay(img, heat.astype(np.float16), boxes, out_path=OUT / "SYNTHETIC_overlay_full.png",
                                   case_id="c_0demo000"))
    panel_w, panel_h = 683, 768
    with Image.open(full) as im:
        f = min(panel_w / im.width, panel_h / im.height)
        fitted = im.convert("RGB").resize((round(im.width * f), round(im.height * f)), Image.Resampling.LANCZOS)
    dash = Image.new("RGB", (1366, 768), (238, 238, 238))
    dash.paste(fitted, ((panel_w - fitted.width) // 2, (panel_h - fitted.height) // 2))
    d = ImageDraw.Draw(dash)
    d.text((panel_w + 30, 30), "dashboard right panel (P1): worklist / case details", font=overlay.font(22),
           fill=(60, 60, 60))
    d.text((panel_w + 30, 70), f"overlay scaled x{f:.2f} to fit 683x768", font=overlay.font(18), fill=(90, 90, 90))
    dash.save(OUT / "SYNTHETIC_dashboard_1366x768.png")
    print(f"wrote {full}\nwrote {OUT / 'SYNTHETIC_dashboard_1366x768.png'} (overlay scale {f:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
