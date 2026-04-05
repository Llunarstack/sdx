"""Tests for config.defaults.art_mediums guidance packs."""

from config.defaults.art_mediums import detect_medium_ids, guidance_fragments


def test_detect_digital_medium():
    ids = detect_medium_ids("digital painting portrait in photoshop", include_photography=True)
    assert "digital_painting" in ids


def test_detect_photo_toggle():
    ids_on = detect_medium_ids("street photography candid portrait photo", include_photography=True)
    assert any(i.endswith("_photo") or "street_documentary_photo" == i for i in ids_on)
    ids_off = detect_medium_ids("street photography candid portrait photo", include_photography=False)
    assert "street_documentary_photo" not in ids_off


def test_guidance_auto_adds_anatomy_lite_for_people():
    pos, neg = guidance_fragments(
        "woman portrait, digital painting",
        "auto",
        include_photography=True,
        anatomy_mode="lite",
    )
    assert "accurate anatomy" in pos.lower()
    assert "bad anatomy" in neg.lower()


def test_guidance_all_contains_multiple_mediums():
    pos, neg = guidance_fragments("landscape", "all", include_photography=False, anatomy_mode="none")
    assert len(pos) > 120
    assert len(neg) > 120

