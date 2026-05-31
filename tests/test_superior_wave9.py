"""Wave 9: ZeResFDG, CFG-Zero*, QSilk micrograin."""

from __future__ import annotations

import torch
from utils.generation.cfg_batched import combine_cfg_outputs
from utils.generation.cfg_zero_star import cfg_zero_optimized_scale, cfg_zero_star_combine, zero_init_step_count
from utils.generation.guidance_stack import combine_guided_prediction
from utils.generation.micrograin_stabilizer import qsilk_micrograin_stabilize
from utils.generation.zeresfdg import (
    SpectralGuidanceEMA,
    apply_zeresfdg_cfg,
    energy_rescale_guided,
    zero_project_delta,
)


def test_zero_project_delta() -> None:
    uncond = torch.tensor([[[[1.0, 0.0]]]])
    delta = torch.tensor([[[[1.0, 1.0]]]])
    r = zero_project_delta(delta, uncond)
    dot = float((r * uncond).sum())
    assert abs(dot) < 1e-5


def test_zeresfdg_differs_from_cfg() -> None:
    cond = torch.randn(1, 4, 16, 16)
    uncond = torch.randn(1, 4, 16, 16)
    std = uncond + 7.0 * (cond - uncond)
    zr = apply_zeresfdg_cfg(cond, uncond, cfg_scale=7.0, cfg_rescale=0.7, strength=1.0)
    assert not torch.allclose(std, zr)


def test_spectral_ema_high_scale() -> None:
    ema = SpectralGuidanceEMA(detail_threshold=0.01)
    lat = torch.randn(1, 4, 8, 8)
    for _ in range(20):
        hs = ema.update(lat)
    assert hs >= ema.conservative_high_scale


def test_cfg_zero_star_zero_init() -> None:
    cond = torch.ones(1, 2, 2, 2)
    uncond = torch.zeros(1, 2, 2, 2)
    out = cfg_zero_star_combine(cond, uncond, cfg_scale=7.0, sample_step=0, total_steps=50)
    assert float(out.abs().max()) < 1e-6


def test_cfg_zero_optimized_scale() -> None:
    cond = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)
    uncond = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
    st = cfg_zero_optimized_scale(cond, uncond)
    assert float(st.mean()) > 0.5


def test_combine_guided_zeresfdg_priority() -> None:
    cond = torch.randn(1, 4, 8, 8)
    uncond = torch.randn(1, 4, 8, 8)
    out = combine_guided_prediction(
        cond,
        uncond,
        uncond,
        cfg_scale=5.0,
        zeresfdg_strength=1.0,
        fdg_strength=0.65,
    )
    assert out.shape == cond.shape


def test_qsilk_stabilize_bounded() -> None:
    x = torch.randn(1, 4, 16, 16) * 5.0
    y = qsilk_micrograin_stabilize(x, detail_amount=0.1)
    assert y.abs().max() <= x.abs().max() + 0.5


def test_zero_init_step_count() -> None:
    assert zero_init_step_count(50, zero_init_frac=0.04) >= 1


def test_energy_rescale() -> None:
    guided = torch.randn(1, 4, 4, 4) * 3.0
    cond = torch.randn(1, 4, 4, 4)
    out = energy_rescale_guided(guided, cond)
    o = out.norm()
    c = cond.norm()
    assert abs(float(o - c) / float(c + 1e-6)) < 0.2


def test_combine_cfg_outputs_cfg_zero() -> None:
    cond = torch.randn(1, 2, 4, 4)
    uncond = torch.randn(1, 2, 4, 4)
    out = combine_cfg_outputs(
        cond,
        uncond,
        uncond,
        cfg_scale=7.0,
        cfg_zero_star=True,
        sample_step=0,
        total_steps=100,
    )
    assert float(out.abs().max()) < 1e-5
