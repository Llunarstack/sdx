"""Caption helpers: multi-person anti-blending detection."""

from data.caption_utils import ANTI_BLEND_NEGATIVE, ANTI_BLEND_POSITIVE, add_anti_blending_and_count


def test_single_character_caption_gets_no_anti_blending():
    assert add_anti_blending_and_count("1girl, solo, smile", "blurry") == ("1girl, solo, smile", "blurry")
    assert add_anti_blending_and_count("1 person walking a dog", "") == ("1 person walking a dog", "")


def test_several_people_get_anti_blending():
    for caption in ("1girl, 1boy, park", "2girls, beach", "3 people at a table", "a crowd at night"):
        cap, neg = add_anti_blending_and_count(caption, "blurry")
        assert cap == f"{caption}, {ANTI_BLEND_POSITIVE}"
        assert neg == f"blurry, {ANTI_BLEND_NEGATIVE}"
