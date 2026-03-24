import numpy as np

from utils.quality.test_time_pick import (
    pick_best_indices,
    score_color_cast_neutrality,
    score_tiling_artifact_free,
)


def test_score_tiling_artifact_free_range():
    img = np.full((64, 64, 3), 127, dtype=np.uint8)
    s = score_tiling_artifact_free(img)
    assert 0.0 <= s <= 1.0


def test_pick_best_combo_structural_runs():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    idx, scores = pick_best_indices([a, b], "a subject", "combo_structural", "cpu")
    assert idx in (0, 1)
    assert len(scores) == 2


def test_score_color_cast_neutrality_range():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 255
    s = score_color_cast_neutrality(img)
    assert 0.0 <= s <= 1.0


def test_pick_best_combo_hq_runs():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 200, dtype=np.uint8)
    idx, scores = pick_best_indices([a, b], "a subject", "combo_hq", "cpu")
    assert idx in (0, 1)
    assert len(scores) == 2
