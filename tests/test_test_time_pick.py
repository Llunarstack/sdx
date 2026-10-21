import numpy as np
import pytest
from utils.quality import test_time_pick


def test_infer_expected_people_count_from_prompt_patterns():
    assert test_time_pick.infer_expected_people_count("exactly 3 people in a cafe scene") == 3
    assert test_time_pick.infer_expected_people_count("two characters, dynamic action pose") == 2
    assert test_time_pick.infer_expected_people_count("portrait of 1girl, studio light") == 1
    assert test_time_pick.infer_expected_people_count("beautiful landscape, no people") == 0


def test_infer_expected_object_count_from_prompt_patterns():
    n, obj = test_time_pick.infer_expected_object_count("exactly 7 coins on a wooden table")
    assert n == 7
    assert obj == "coin"
    n2, obj2 = test_time_pick.infer_expected_object_count("two candles in a dark room")
    assert n2 == 2
    assert obj2 == "candle"


def test_score_people_count_match_uses_error_ratio(monkeypatch):
    monkeypatch.setattr(test_time_pick, "estimate_people_count", lambda _im: 3)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert test_time_pick.score_people_count_match(img, 3) == 1.0
    assert abs(test_time_pick.score_people_count_match(img, 2) - 0.5) < 1e-8


def test_combo_count_prefers_matching_candidate(monkeypatch):
    # Make CLIP/edge neutral so count score dominates in this unit test.
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_edge_sharpness", lambda im: 1.0)
    monkeypatch.setattr(
        test_time_pick,
        "score_people_count_match",
        lambda im, exp: 1.0 if int(im[0, 0, 0]) == int(exp) else 0.0,
    )

    img_wrong = np.zeros((8, 8, 3), dtype=np.uint8)  # marker 0
    img_right = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right[0, 0, 0] = 2  # marker 2

    best, scores = test_time_pick.pick_best_indices(
        [img_wrong, img_right],
        "exactly 2 people",
        "combo_count",
        "cpu",
    )
    assert best == 1
    assert scores[1] > scores[0]


def test_combo_count_object_mode_prefers_matching_candidate(monkeypatch):
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_edge_sharpness", lambda im: 1.0)
    monkeypatch.setattr(
        test_time_pick,
        "score_object_count_match",
        lambda im, exp, object_hint="": 1.0 if int(im[0, 0, 0]) == int(exp) else 0.0,
    )

    img_wrong = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right[0, 0, 0] = 5

    best, scores = test_time_pick.pick_best_indices(
        [img_wrong, img_right],
        "five candles on a cake",
        "combo_count",
        "cpu",
        expected_count_target="objects",
        expected_count_object="candle",
    )
    assert best == 1
    assert scores[1] > scores[0]


def test_rank_aggregation_is_invariant_to_metric_scale():
    a = [0.1, 0.5, 0.3]
    b = [3.0, 1.0, 2.0]
    base = test_time_pick.rank_aggregate({"a": a, "b": b}, {"a": 0.7, "b": 0.3})
    scaled = test_time_pick.rank_aggregate(
        {"a": [x * 1e6 for x in a], "b": [np.log(x) for x in b]}, {"a": 0.7, "b": 0.3}
    )
    assert scaled == pytest.approx(base)
    assert int(np.argmax(base)) == 1
    # Float noise ties instead of ordering candidates; a missing term drops out.
    assert test_time_pick.rank01([1.0, 1.0 + 1e-12, 0.5]) == pytest.approx([0.75, 0.75, 0.0])
    assert test_time_pick.rank_aggregate({"a": a, "b": None}, {"a": 1.0, "b": 1.0}) == pytest.approx([0.0, 1.0, 0.5])


def test_combo_realism_prefers_candidate_with_better_realism_metrics(monkeypatch):
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_photographic_detail_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_exposure_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_dynamic_range_headroom", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_color_cast_neutrality", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_skin_tone_naturalness", lambda im: float(im[0, 0, 0]) / 255.0)

    img_low = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high[0, 0, 0] = 255

    best, scores = test_time_pick.pick_best_indices([img_low, img_high], "photo portrait", "combo_realism", "cpu")
    assert best == 1
    assert scores[1] > scores[0]


def test_score_aesthetic_proxy_is_bounded():
    flat = np.full((8, 8, 3), 127, dtype=np.uint8)
    s = test_time_pick.score_aesthetic_proxy(flat)
    assert 0.0 <= s <= 1.0


def test_aesthetic_metric_prefers_higher_proxy(monkeypatch):
    monkeypatch.setattr(test_time_pick, "score_aesthetic_proxy", lambda im: float(im[0, 0, 0]) / 255.0)
    low = np.zeros((4, 4, 3), dtype=np.uint8)
    high = np.full((4, 4, 3), 255, dtype=np.uint8)
    best, scores = test_time_pick.pick_best_indices([low, high], "x", "aesthetic", "cpu")
    assert best == 1
    assert scores[1] > scores[0]


def test_combo_count_vit_still_prefers_count_when_vit_is_neutral(monkeypatch):
    monkeypatch.setattr(
        test_time_pick, "score_clip_similarity", lambda imgs, prompt, device, model_id: [0.5 for _ in imgs]
    )
    monkeypatch.setattr(test_time_pick, "score_edge_sharpness", lambda im: 1.0)
    monkeypatch.setattr(
        test_time_pick,
        "score_people_count_match",
        lambda im, exp: 1.0 if int(im[0, 0, 0]) == int(exp) else 0.0,
    )
    img_wrong = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right = np.zeros((8, 8, 3), dtype=np.uint8)
    img_right[0, 0, 0] = 2
    best, scores = test_time_pick.pick_best_indices(
        [img_wrong, img_right], "exactly 2 people", "combo_count_vit", "cpu"
    )
    assert best == 1
    assert scores[1] > scores[0]


def test_aesthetic_realism_prefers_candidate_with_better_realism_metrics(monkeypatch):
    monkeypatch.setattr(test_time_pick, "score_photographic_detail_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_exposure_balance", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_dynamic_range_headroom", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_color_cast_neutrality", lambda im: float(im[0, 0, 0]) / 255.0)
    monkeypatch.setattr(test_time_pick, "score_skin_tone_naturalness", lambda im: float(im[0, 0, 0]) / 255.0)

    img_low = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high = np.zeros((8, 8, 3), dtype=np.uint8)
    img_high[0, 0, 0] = 255

    best, scores = test_time_pick.pick_best_indices([img_low, img_high], "photo portrait", "aesthetic_realism", "cpu")
    assert best == 1
    assert scores[1] > scores[0]


def test_vit_views_keep_aspect_ratio():
    from PIL import Image

    wide = test_time_pick._vit_views(Image.new("RGB", (600, 400)), 224)
    assert len(wide) == 3 and all(v.size == (224, 224) for v in wide)
    assert len(test_time_pick._vit_views(Image.new("RGB", (300, 300)), 224)) == 1


def test_color_cast_magnitude():
    gray = np.full((64, 64, 3), 128.0, dtype=np.float32)
    warm = np.clip(gray + np.array([30.0, 20.0, -35.0]), 0, 255).astype(np.uint8)
    assert test_time_pick.color_cast_magnitude(warm) > 0.06
    neutral = np.random.default_rng(0).integers(90, 170, size=(64, 64, 3), dtype=np.uint8)
    assert test_time_pick.color_cast_magnitude(neutral) < 0.06
    assert test_time_pick.color_cast_magnitude(np.zeros((16, 16, 3), dtype=np.uint8)) == 0.0
    assert test_time_pick.color_cast_magnitude(np.full((16, 16, 3), 255, dtype=np.uint8)) == 0.0
