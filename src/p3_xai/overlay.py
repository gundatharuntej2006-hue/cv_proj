"""P3-11: overlay renderer for R3 render_overlay (X-ray + Grad-CAM + detector boxes + caption).

Layout (1024 x 1204 px; the 512 image is shown at 2x):
    header   case id (only the case_id, never a patient identifier) and what is shown
    image    grey X-ray, 'inferno' heatmap with opacity growing with attention, detector boxes
             with score labels
    legend   colour bar "low -> high attention"
    caption  "Attention evidence, not a finding", burned into the pixels on a dark-red band

Sized so that, scaled to fit a half-width panel of the 1366x768 dashboard (683x768, factor ~0.64),
the caption is ~18 px and box labels ~13 px tall.

Boxes: a DetResult draws only its boxes above the locked threshold (the first n_lesions_above_thr,
since boxes are sorted by score); a list (CaseResult [{xyxy, score}] or rows [x0,y0,x1,y1(,score)])
is drawn as given. No boxes / None -> heatmap only (detector panel omitted, no stray boxes).
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

CAPTION = "Attention evidence, not a finding"
LEGEND = "low  ->  high attention"   # ASCII: Pillow's built-in font has no arrow glyph
SCALE = 2
IMG = 512 * SCALE
HEADER_H, LEGEND_H, CAPTION_H = 56, 64, 72
WIDTH, HEIGHT = IMG, HEADER_H + IMG + LEGEND_H + CAPTION_H
CAPTION_FONT, LABEL_FONT, TEXT_FONT = 34, 24, 26
MAX_ALPHA = 0.55
BOX_RGB, CAPTION_BG = (0, 230, 118), (120, 0, 0)
CASE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=1)
def colormap() -> np.ndarray:
    """256 x 3 uint8 'inferno' (perceptually uniform, readable for colour-blind viewers)."""
    from matplotlib import colormaps
    return (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def _boxes(boxes) -> list[tuple[list[float], float | None]]:
    if boxes is None:
        return []
    if hasattr(boxes, "boxes_full512"):                      # DetResult: boxes above the threshold only
        n = int(boxes.n_lesions_above_thr)
        return [(list(map(float, b)), float(s)) for b, s in zip(boxes.boxes_full512[:n], boxes.scores[:n])]
    out = []
    for b in boxes:
        if isinstance(b, dict):
            out.append(([float(v) for v in b["xyxy"]], None if b.get("score") is None else float(b["score"])))
        else:
            v = [float(x) for x in b]
            out.append((v[:4], v[4] if len(v) > 4 else None))
    return out


def _gray(img512) -> np.ndarray:
    img = np.asarray(img512)
    if img.shape[:2] != (512, 512):
        raise ValueError(f"img512 must be 512x512, got {img.shape}")
    if img.ndim == 3:
        img = np.asarray(Image.fromarray(np.ascontiguousarray(img[..., :3]).astype(np.uint8)).convert("L"))
    return img.astype(np.uint8)


def _heat(heatmap) -> np.ndarray:
    if heatmap is None:
        return np.zeros((512, 512), np.float32)
    h = np.asarray(heatmap, np.float32)
    if h.shape != (512, 512):
        raise ValueError(f"heatmap must be [512, 512], got {h.shape}")
    return np.clip(np.nan_to_num(h), 0.0, 1.0)


def draw_caption(canvas: Image.Image) -> None:
    """The caption band; part of the pixels, so it cannot be stripped like metadata."""
    d = ImageDraw.Draw(canvas)
    top = HEIGHT - CAPTION_H
    d.rectangle([0, top, WIDTH, HEIGHT], fill=CAPTION_BG)
    f = font(CAPTION_FONT)
    x0, y0, x1, y1 = d.textbbox((0, 0), CAPTION, font=f)
    d.text(((WIDTH - (x1 - x0)) // 2 - x0, top + (CAPTION_H - (y1 - y0)) // 2 - y0), CAPTION, font=f, fill="white")


def caption_reference() -> Image.Image:
    """The caption band alone, rendered the same way (used to verify it is present in an overlay)."""
    ref = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw_caption(ref)
    return ref.crop((0, HEIGHT - CAPTION_H, WIDTH, HEIGHT))


def render(img512, heatmap, boxes, case_id: str | None = None) -> Image.Image:
    gray = np.repeat(_gray(img512)[..., None], 3, axis=2).astype(np.float32)
    h = _heat(heatmap)
    rgb = colormap()[(h * 255).round().astype(np.uint8)].astype(np.float32)
    a = (MAX_ALPHA * h)[..., None]
    blended = (gray * (1 - a) + rgb * a).clip(0, 255).astype(np.uint8)
    base = Image.fromarray(blended).resize((IMG, IMG), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (20, 20, 20))
    canvas.paste(base, (0, HEADER_H))
    d = ImageDraw.Draw(canvas)
    drawn = _boxes(boxes)
    title = "Grad-CAM attention (TB output)" + ("  +  detector boxes" if drawn else "")
    d.text((16, 14), title, font=font(TEXT_FONT), fill="white")
    if case_id:
        f = font(TEXT_FONT)
        w = d.textlength(f"case {case_id}", font=f)
        d.text((WIDTH - 16 - w, 14), f"case {case_id}", font=f, fill=(200, 200, 200))

    lf = font(LABEL_FONT)
    for (x0, y0, x1, y1), score in drawn:
        X0, Y0, X1, Y1 = (min(max(v * SCALE, 0), IMG - 1) for v in (x0, y0, x1, y1))
        d.rectangle([X0, HEADER_H + Y0, X1, HEADER_H + Y1], outline=BOX_RGB, width=3)
        if score is not None:
            label = f"{score:.2f}"
            tx0, ty0, tx1, ty1 = d.textbbox((0, 0), label, font=lf)
            tw, th = tx1 - tx0 + 8, ty1 - ty0 + 6
            lx = min(max(X0, 0), IMG - tw)
            ly = HEADER_H + (Y0 - th if Y0 - th >= 0 else Y1)      # above the box, else below
            ly = min(ly, HEADER_H + IMG - th)
            d.rectangle([lx, ly, lx + tw, ly + th], fill=BOX_RGB)
            d.text((lx + 4 - tx0, ly + 3 - ty0), label, font=lf, fill="black")

    top = HEADER_H + IMG
    bar_x0, bar_x1, bar_y0, bar_y1 = 240, WIDTH - 240, top + 12, top + 34
    bar = colormap()[np.linspace(0, 255, bar_x1 - bar_x0).astype(np.uint8)]
    canvas.paste(Image.fromarray(np.repeat(bar[None], bar_y1 - bar_y0, axis=0)), (bar_x0, bar_y0))
    f = font(TEXT_FONT)
    d.text((16, top + 10), "attention", font=f, fill="white")
    lw = d.textlength(LEGEND, font=font(LABEL_FONT))
    d.text(((WIDTH - lw) / 2, bar_y1 + 2), LEGEND, font=font(LABEL_FONT), fill="white")
    draw_caption(canvas)
    return canvas


def default_path(runs_root: Path, run_id: str, case_id: str | None, digest: str) -> Path:
    name = f"{case_id}_overlay.png" if case_id else f"overlay_{digest}.png"
    return runs_root / run_id / name


def save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = PngInfo()
    meta.add_text("Caption", CAPTION)
    meta.add_text("Producer", "p3_xai.api.render_overlay")
    tmp = path.with_suffix(".png.tmp")
    img.save(tmp, format="PNG", pnginfo=meta)
    tmp.replace(path)
    return path


def digest_of(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]
