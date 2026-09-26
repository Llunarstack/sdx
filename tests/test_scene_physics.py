import numpy as np
from utils.quality.scene_physics import (
    prompt_wants_occlusion,
    prompt_wants_reflection,
    score_occlusion,
    score_reflection,
)


def _fence_rgb(size: int = 128) -> np.ndarray:
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(0, size, 4):
        rgb[:, x : x + 2] = 255
    return rgb


def _uniform_blob_rgb(size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 24, dtype=np.uint8)
    rgb[12 : size - 12, 12 : size - 12] = 180
    return rgb


def _darker_flip_reflection(size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 36, dtype=np.uint8)
    half = size // 2
    # Asymmetric blobs near the top of the upper half so a flip is distinctive.
    rgb[6:28, 14:58] = 230
    rgb[44:58, 80:110] = 210
    rgb[half : half + half] = (rgb[:half][::-1].astype(np.float32) * 0.42).astype(np.uint8)
    return rgb


def _two_bright_stacked_blobs(size: int = 128) -> np.ndarray:
    rgb = np.full((size, size, 3), 36, dtype=np.uint8)
    half = size // 2
    rgb[6:28, 14:58] = 230
    rgb[44:58, 80:110] = 210
    rgb[half + 6 : half + 28, 14:58] = 230
    rgb[half + 44 : half + 58, 80:110] = 210
    return rgb


def test_score_occlusion_fence_beats_blob():
    fence = _fence_rgb()
    blob = _uniform_blob_rgb()
    assert score_occlusion(fence) > score_occlusion(blob)


def test_score_reflection_darker_flip_beats_equal_blobs():
    reflected = _darker_flip_reflection()
    stacked = _two_bright_stacked_blobs()
    assert score_reflection(reflected) > score_reflection(stacked)


def test_prompt_wants_occlusion():
    assert prompt_wants_occlusion("person behind a chain-link fence") is True
    assert prompt_wants_occlusion("looking through gaps in the railing") is True
    assert prompt_wants_occlusion("cat hidden behind a sofa") is True
    assert prompt_wants_occlusion("face occluded by leaves") is True
    assert prompt_wants_occlusion("studio portrait of a red apple") is False
    assert prompt_wants_occlusion("") is False


def test_prompt_wants_reflection():
    assert prompt_wants_reflection("mountain reflection in a lake") is True
    assert prompt_wants_reflection("chrome sculpture on granite") is True
    assert prompt_wants_reflection("chrome kettle on a wet table") is True
    assert prompt_wants_reflection("mirror of the sky in a puddle") is True
    assert prompt_wants_reflection("the building is reflected in water") is True
    assert prompt_wants_reflection("hall of mirrors") is False
    assert prompt_wants_reflection("a red apple on a desk") is False
    assert prompt_wants_reflection("") is False


def test_score_spatial_color_bind_prefers_correct_halves():
    from utils.quality.scene_physics import score_spatial_color_bind

    prompt = "a red cube to the left of a blue sphere"
    good = np.zeros((64, 64, 3), dtype=np.uint8)
    good[:, :32] = (200, 20, 20)
    good[:, 32:] = (20, 30, 200)
    bad = np.zeros((64, 64, 3), dtype=np.uint8)
    bad[:, :32] = (20, 30, 200)
    bad[:, 32:] = (200, 20, 20)
    assert score_spatial_color_bind(good, prompt) > score_spatial_color_bind(bad, prompt)
    assert score_spatial_color_bind(good, "a cat sitting in a garden") is None
