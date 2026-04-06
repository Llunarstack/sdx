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


def test_detect_new_3d_and_photo_mediums():
    ids_3d = detect_medium_ids("archviz octane render interior visualization", include_photography=True)
    assert "archviz_3d" in ids_3d
    ids_photo = detect_medium_ids("wedding event photo ceremony portrait", include_photography=True)
    assert "wedding_event_photo" in ids_photo


def test_detect_medium_ids_with_aliases():
    ids = detect_medium_ids("bw film photo and toon-shaded 3d robot scene", include_photography=True)
    assert "film_bw_photo" in ids
    assert "toon_3d" in ids


def test_detect_new_storyboard_and_diorama_aliases():
    ids = detect_medium_ids("animatic storyboard for action beats with miniature photo lighting", include_photography=True)
    assert "storyboard_sketch" in ids
    assert "miniature_diorama_photo" in ids

