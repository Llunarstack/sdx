"""Tests for ``research.visual_quality`` (CPU)."""

from __future__ import annotations

import torch
from research.visual_quality import perceptual_proxies as pp
from research.visual_quality import rank_and_gate as rg


def _rgb_random(b: int = 2, h: int = 32, w: int = 32) -> torch.Tensor:
    return torch.rand(b, 3, h, w)


def test_laplacian_sharpness_blur_lower_than_sharp() -> None:
    x = _rgb_random(1, 64, 64)
    y = torch.nn.functional.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    s_x = float(pp.laplacian_sharpness(x)[0].item())
    s_y = float(pp.laplacian_sharpness(y)[0].item())
    assert s_x > s_y


def test_exposure_naturalness_mid_grey_best() -> None:
    mid = torch.full((1, 3, 16, 16), 0.5)
    dark = torch.zeros(1, 3, 16, 16)
    e_mid = float(pp.exposure_naturalness(mid)[0].item())
    e_dark = float(pp.exposure_naturalness(dark)[0].item())
    assert e_mid > e_dark


def test_combined_quality_proxy_shape() -> None:
    x = _rgb_random(4, 24, 24)
    c = pp.combined_quality_proxy(x)
    assert c.shape == (4,)


def test_rank_samples_by_proxy() -> None:
    x = _rgb_random(3, 16, 16)
    r = rg.rank_samples_by_proxy(x)
    assert len(r) == 3
    assert {i for i, _ in r} == {0, 1, 2}
    scores = [s for _, s in r]
    assert scores == sorted(scores, reverse=True)


def test_best_of_n_index() -> None:
    dull = torch.full((1, 3, 8, 8), 0.5)
    sharp = torch.rand(1, 3, 8, 8)
    idx = rg.best_of_n_index([dull, sharp])
    assert idx == 1
