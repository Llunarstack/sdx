import numpy as np
from utils.generation import glyph_canvas
from utils.quality import contact_shadow


def test_render_glyph_canvas_shop_sign():
    img = glyph_canvas.render_glyph_canvas('shop sign that says "OPEN"')
    assert img is not None
    assert img.size == (512, 512)
    arr = np.asarray(img)
    assert arr.min() < 250
    assert np.any(arr < 200)


def test_render_glyph_canvas_bland_prompt_returns_none():
    assert glyph_canvas.render_glyph_canvas("sunset over mountains") is None


def test_extract_glyph_strings_quoted_and_bracketed():
    strings = glyph_canvas.extract_glyph_strings('poster with [text: SALE] and "50% OFF"')
    assert "SALE" in strings
    assert "50% OFF" in strings


def _synthetic_feet_with_contact(*, contact: bool, size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 190, dtype=np.uint8)
    y0 = int(size * 0.78)
    foot_y = y0 + int(size * 0.10)
    foot_x0 = size // 3
    foot_x1 = 2 * size // 3
    rgb[foot_y : foot_y + 4, foot_x0:foot_x1] = 40
    if contact:
        # Subtle AO just below feet; stays outside subject mask (|175-190| <= 18).
        rgb[foot_y + 4 : foot_y + 6, foot_x0:foot_x1] = 175
    return rgb


def test_contact_shadow_scores_grounded_higher_than_floating():
    grounded = _synthetic_feet_with_contact(contact=True)
    floating = _synthetic_feet_with_contact(contact=False)
    assert contact_shadow.score_contact_shadow(grounded) > contact_shadow.score_contact_shadow(floating)


def test_apply_contact_shadow_darkens_lower_band():
    rgb = _synthetic_feet_with_contact(contact=False)
    y0 = int(rgb.shape[0] * 0.78)
    before = float(rgb[y0:, :].mean())
    out = contact_shadow.apply_contact_shadow(rgb, strength=0.45)
    after = float(out[y0:, :].mean())
    assert after < before


def test_prompt_wants_ground_contact():
    assert contact_shadow.prompt_wants_ground_contact("woman standing on wet pavement") is True
    assert contact_shadow.prompt_wants_ground_contact("macro of an eye") is False
