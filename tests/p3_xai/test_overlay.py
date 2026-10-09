"""P3-11: overlay renderer (R3 render_overlay)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from p3_det.result import DetResult  # noqa: E402
from p3_xai import api, overlay  # noqa: E402

CROP = [40, 20, 470, 490]
PANEL = (683, 768)          # half of the 1366x768 dashboard


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TB_OVERLAY_DIR", raising=False)
    monkeypatch.setenv("TB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("TB_RUN_ID", "r0001")
    return tmp_path / "runs"


@pytest.fixture
def img():
    yy, xx = np.mgrid[0:512, 0:512]
    return (90 + 60 * np.sin(xx / 40) * np.cos(yy / 55)).clip(0, 255).astype(np.uint8)


@pytest.fixture
def heat():
    yy, xx = np.mgrid[0:512, 0:512]
    h = np.exp(-((xx - 300) ** 2 + (yy - 200) ** 2) / (2 * 40 ** 2)).astype(np.float32)
    out = np.zeros_like(h)
    out[CROP[1]:CROP[3], CROP[0]:CROP[2]] = h[CROP[1]:CROP[3], CROP[0]:CROP[2]]
    return (out / out.max()).astype(np.float16)


def image_area(im: Image.Image) -> np.ndarray:
    a = np.asarray(im.convert("RGB"))
    return a[overlay.HEADER_H:overlay.HEADER_H + overlay.IMG]


def box_pixels(im: Image.Image) -> int:
    return int((image_area(im) == overlay.BOX_RGB).all(axis=2).sum())


# --- caption ---------------------------------------------------------------------------------------

def test_caption_drawn_by_the_renderer(img, heat, monkeypatch):
    texts = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **kw):
        texts.append(text)
        return real(self, xy, text, *a, **kw)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    overlay.render(img, heat, [])
    assert "Attention evidence, not a finding" in texts


@pytest.mark.parametrize("boxes", [[], None, [[300, 180, 360, 240, 0.81]]], ids=["empty", "none", "one"])
def test_caption_present_in_the_pixels(img, heat, boxes):
    path = api.render_overlay(img, heat, boxes)
    with Image.open(path) as im:
        band = im.convert("RGB").crop((0, overlay.HEIGHT - overlay.CAPTION_H, overlay.WIDTH, overlay.HEIGHT))
        assert np.array_equal(np.asarray(band), np.asarray(overlay.caption_reference()))  # exact text pixels
        assert (np.asarray(band) == 255).all(axis=2).sum() > 500                          # white glyphs exist
        assert im.text["Caption"] == "Attention evidence, not a finding"


# --- boxes ------------------------------------------------------------------------------------------

def test_zero_boxes_draws_no_boxes_and_header_says_heatmap_only(img, heat, monkeypatch):
    texts = []
    real = ImageDraw.ImageDraw.text
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", lambda self, xy, t, *a, **k: (texts.append(t), real(self, xy, t, *a, **k))[1])
    for boxes in ([], None, DetResult(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0)):
        with Image.open(api.render_overlay(img, heat, boxes)) as im:
            assert box_pixels(im) == 0
    assert not any("detector" in t for t in texts)


def test_one_box_with_score_label(img, heat):
    with Image.open(api.render_overlay(img, heat, [{"xyxy": [301, 88, 372, 160], "score": 0.77}])) as im:
        a = image_area(im)
        assert (a[2 * 120, 2 * 301 + 1] == overlay.BOX_RGB).all()          # left edge of the box at 2x
        assert box_pixels(im) > 400                                         # outline + label background


def test_many_boxes(img, heat):
    r = np.random.default_rng(0)
    boxes = []
    for _ in range(60):
        x, y = r.uniform(0, 470, 2)
        boxes.append([x, y, x + r.uniform(5, 40), y + r.uniform(5, 40), float(r.uniform(0.01, 1))])
    boxes.append([500, 500, 512, 512, 0.5])                                  # touching the corner
    with Image.open(api.render_overlay(img, heat, boxes)) as im:
        assert im.size == (overlay.WIDTH, overlay.HEIGHT) and box_pixels(im) > 0


def test_detresult_draws_only_boxes_above_threshold(img, heat):
    det = DetResult(np.array([[100, 100, 150, 150], [300, 300, 350, 350]], np.float32),
                    np.array([0.9, 0.05], np.float32), n_lesions_above_thr=1)
    with Image.open(api.render_overlay(img, heat, det)) as im:
        a = image_area(im)
        assert (a[2 * 125, 2 * 100 + 1] == overlay.BOX_RGB).all()          # above threshold: drawn
        assert not (a[2 * 325, 2 * 300 + 1] == overlay.BOX_RGB).all()      # below threshold: not drawn


# --- heatmap ---------------------------------------------------------------------------------------

def test_zero_heatmap_leaves_xray_uncoloured(img):
    with Image.open(api.render_overlay(img, np.zeros((512, 512), np.float16), [])) as im:
        a = image_area(im).astype(int)
        assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()   # grey only
    with Image.open(api.render_overlay(img, None, [])) as im:
        assert im.size == (overlay.WIDTH, overlay.HEIGHT)


def test_heatmap_colours_only_where_attention(img, heat):
    with Image.open(api.render_overlay(img, heat, [])) as im:
        a = image_area(im).astype(int)
        hot = a[2 * 200, 2 * 300]                                           # heat peak
        cold = a[2 * 500, 2 * 10]                                           # outside the crop: zero heat
        assert len(set(hot)) > 1 and len(set(cold)) == 1


def test_rgb_input_and_bad_shapes(img, heat):
    api.render_overlay(np.stack([img] * 3, -1), heat, [])
    with pytest.raises(ValueError):
        api.render_overlay(img[:256], heat, [])
    with pytest.raises(ValueError):
        api.render_overlay(img, heat[:100], [])


# --- output path and identifiers ---------------------------------------------------------------------

def test_generated_path_under_runs_uses_case_id_only(img, heat, runs_dir):
    p = Path(api.render_overlay(img, heat, [], case_id="c_7f3a9e21"))
    assert p == runs_dir / "r0001" / "c_7f3a9e21_overlay.png" and p.exists()
    with Image.open(p) as im:
        assert im.size == (overlay.WIDTH, overlay.HEIGHT) and im.format == "PNG"
        assert set(im.text) == {"Caption", "Producer"}                      # no other metadata


def test_caller_path_and_refused_identifiers(img, heat, tmp_path):
    target = tmp_path / "custom" / "x.png"
    assert api.render_overlay(img, heat, [], out_path=target) == str(target) and target.exists()
    for bad in ("John Doe", "../../etc/passwd", "MRN 123/45"):
        with pytest.raises(ValueError):
            api.render_overlay(img, heat, [], case_id=bad)


def test_header_text_has_only_allowed_strings(img, heat, monkeypatch):
    texts = []
    real = ImageDraw.ImageDraw.text
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", lambda self, xy, t, *a, **k: (texts.append(t), real(self, xy, t, *a, **k))[1])
    overlay.render(img, heat, [[300, 180, 360, 240, 0.81]], case_id="c_7f3a9e21")
    allowed = {"Grad-CAM attention (TB output)  +  detector boxes", "case c_7f3a9e21", "0.81", "attention",
               overlay.LEGEND, overlay.CAPTION}
    assert set(texts) == allowed


# --- legibility at the dashboard size ------------------------------------------------------------------

def test_legible_when_fitted_to_half_of_1366x768(img, heat, tmp_path):
    path = api.render_overlay(img, heat, [[300, 180, 360, 240, 0.81]])
    with Image.open(path) as im:
        f = min(PANEL[0] / im.width, PANEL[1] / im.height)
        preview = im.resize((round(im.width * f), round(im.height * f)), Image.Resampling.LANCZOS)
    preview.save(tmp_path / "preview.png")
    assert preview.width <= PANEL[0] and preview.height <= PANEL[1]
    cap = overlay.font(overlay.CAPTION_FONT).getbbox(overlay.CAPTION)
    lab = overlay.font(overlay.LABEL_FONT).getbbox("0.81")
    assert (cap[3] - cap[1]) * f >= 14          # caption cap-height after scaling (px)
    assert (lab[3] - lab[1]) * f >= 9           # score digits after scaling (px)
    assert (cap[2] - cap[0]) * f <= preview.width   # caption not cut off


def test_all_drawn_text_is_ascii(img, heat, monkeypatch):
    """Pillow's built-in font lacks e.g. arrows; non-ASCII would render as missing-glyph boxes."""
    texts = []
    real = ImageDraw.ImageDraw.text
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", lambda self, xy, t, *a, **k: (texts.append(t), real(self, xy, t, *a, **k))[1])
    overlay.render(img, heat, [[300, 180, 360, 240, 0.81]], case_id="c_7f3a9e21")
    assert all(t.isascii() for t in texts)
