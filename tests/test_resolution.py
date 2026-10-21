"""Resolution / aspect-ratio correctness: any-grid DiTs, old checkpoints, SD3 shift, canvas, rectangular sampling."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from diffusion.gaussian_diffusion import create_diffusion
from diffusion.pixel_perfect import sampling_canvas_hw
from diffusion.schedules import (
    flux_resolution_shift,
    inverse_time_shift,
    sd3_resolution_shift,
    time_shift,
    vp_shifted_timestep,
)
from diffusion.solvers import build_flow_s_grid
from models.ar_masks_extended import create_block_causal_mask_2d
from models.dit import get_2d_sincos_pos_embed, grid_pos_embed
from models.dit_text import DiT_Predecessor_Text, DiT_Supreme_Text, DiT_Text, grid_ar_mask
from utils.generation.sample_helpers import renoise_for_grid

TINY = dict(
    input_size=8,
    patch_size=2,
    in_channels=4,
    hidden_size=32,
    depth=2,
    num_heads=4,
    text_dim=16,
    learn_sigma=False,
    use_xformers=False,
)


def _tiny(cls=DiT_Text, seed: int = 0, **kw):
    torch.manual_seed(seed)
    m = cls(**TINY, **kw).eval()
    with torch.no_grad():  # adaLN-Zero / zero output layers would make every output identical
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(0.0, 0.2)
    return m


def _run(m, h: int, w: int, b: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, 4, h, w, generator=g)
    txt = torch.randn(b, 5, 16, generator=g)
    with torch.no_grad():
        return m(x, torch.tensor([10, 700])[:b], encoder_hidden_states=txt)


@pytest.mark.parametrize(
    "cls,kw",
    [
        (DiT_Text, {}),
        (DiT_Text, {"use_rope": True, "num_register_tokens": 2}),
        (DiT_Text, {"num_ar_blocks": 2}),
        (DiT_Text, {"use_rope": True, "kv_merge_factor": 2, "num_register_tokens": 1}),
        (DiT_Text, {"uvit_skips": True, "mmdit_every_n": 1, "size_embed_dim": 8}),
        (DiT_Predecessor_Text, {"num_ar_blocks": 2}),
        (DiT_Supreme_Text, {"use_rope": True}),
    ],
)
def test_any_patch_grid_forward(cls, kw):
    m = _tiny(cls, **kw)
    for h, w in ((8, 8), (8, 12), (12, 6), (6, 10)):
        out = _run(m, h, w)
        assert out.shape == (2, 4, h, w)
        assert torch.isfinite(out).all() and out.abs().sum() > 0


def test_old_state_dict_loads_and_native_grid_is_unchanged():
    m = _tiny(use_rope=True, num_ar_blocks=2)
    sd = m.state_dict()
    assert not any("rope" in k for k in sd)  # 2D RoPE is parameter-free
    fresh = DiT_Text(**TINY, use_rope=True, num_ar_blocks=2).eval()
    assert set(fresh.state_dict()) == set(sd)
    fresh.load_state_dict(sd, strict=True)
    for h, w in ((8, 8), (8, 12)):
        assert torch.equal(_run(fresh, h, w), _run(m, h, w))
    # The native grid keeps using the stored table, which is the plain MAE sin/cos table.
    assert grid_pos_embed(m.pos_embed, 4, 4, 4) is m.pos_embed
    ref = torch.from_numpy(get_2d_sincos_pos_embed(32, 4)).float()
    assert torch.allclose(m.pos_embed[0], ref)


def test_pixart_position_interpolation():
    pe = torch.from_numpy(get_2d_sincos_pos_embed(32, 4)).float().unsqueeze(0)
    d = pe.shape[-1]
    up = grid_pos_embed(pe, 4, 8, 8).reshape(8, 8, d)
    assert torch.allclose(up[::2, ::2], pe.reshape(4, 4, d), atol=1e-6)
    wide = grid_pos_embed(pe, 4, 4, 8).reshape(4, 8, d)
    assert torch.allclose(wide[:, ::2], pe.reshape(4, 4, d), atol=1e-6)
    assert grid_pos_embed(pe, 4, 4, 8) is grid_pos_embed(pe, 4, 4, 8)  # cached per shape


def test_block_ar_mask_follows_the_grid():
    m = _tiny(num_ar_blocks=2)
    assert grid_ar_mask(m, 4, 4) is m._ar_mask
    assert torch.equal(grid_ar_mask(m, 3, 5), create_block_causal_mask_2d(3, 5, 2))
    m._ar_mask = None
    assert grid_ar_mask(m, 3, 5) is None


def test_shift_formulas():
    assert time_shift(0.5, 3.0) == pytest.approx(0.75)
    s = np.linspace(0.01, 0.99, 25)
    assert np.allclose(time_shift(s, 1.0), s)
    assert np.allclose(inverse_time_shift(time_shift(s, 2.5), 2.5), s)
    snr = ((1 - s) / s) ** 2
    s3 = time_shift(s, 3.0)
    assert np.allclose(((1 - s3) / s3) ** 2, snr / 9.0)  # SNR divided by shift^2
    assert sd3_resolution_shift(4096, 1024) == pytest.approx(2.0)
    assert sd3_resolution_shift(1024, 1024, base_shift=3.0) == pytest.approx(3.0)
    assert flux_resolution_shift(256) == pytest.approx(math.exp(0.5))
    assert flux_resolution_shift(4096) == pytest.approx(math.exp(1.15))


def test_vp_shifted_timestep_matches_log_snr():
    ac = create_diffusion().alpha_cumprod.numpy().astype(np.float64)
    log_snr = np.log(ac / (1 - ac))
    assert vp_shifted_timestep(ac, 300, 1.0) == 300
    t2 = vp_shifted_timestep(ac, 300, 2.0)
    assert t2 > 300
    assert abs(log_snr[t2] - (log_snr[300] - 2 * math.log(2.0))) < 0.05
    assert vp_shifted_timestep(ac, 300, 0.5) < 300


def test_flow_grid_shift_keeps_endpoints_and_spends_more_time_at_noise():
    base = build_flow_s_grid("linear", 10)
    g = build_flow_s_grid("linear", 10, shift=3.0)
    assert g[0] == 1.0 and g[-1] == 0.0
    assert np.all(np.diff(g) < 0) and np.all(g >= base - 1e-12)
    assert np.allclose(g, time_shift(base, 3.0))
    part = build_flow_s_grid("ays", 8, s_start=0.4, shift=2.0)
    assert part[0] == pytest.approx(0.4) and part[-1] == 0.0


def test_sampling_canvas():
    assert sampling_canvas_hw(512, 512, native_px=512, stride_px=16) == (512, 512)
    assert sampling_canvas_hw(512, 768, native_px=512, stride_px=16) == (512, 768)  # within 2x area
    h, w = sampling_canvas_hw(576, 1024, native_px=512, stride_px=16)  # 2.25x area -> native area, 16:9
    assert (h, w) == (384, 688) and h % 16 == 0 and w % 16 == 0
    assert sampling_canvas_hw(512, 768, native_px=512, stride_px=16, max_area_ratio=1.0) == (416, 624)


def test_sample_loop_on_rectangular_grid(monkeypatch):
    m = _tiny()
    diff = create_diffusion()
    txt = torch.randn(1, 5, 16)
    kw = dict(
        model_kwargs_cond={"encoder_hidden_states": txt},
        model_kwargs_uncond={"encoder_hidden_states": torch.zeros_like(txt)},
        cfg_scale=2.0,
        num_inference_steps=4,
        device="cpu",
    )
    out = diff.sample_loop(m, (1, 4, 8, 12), solver="ddim", scheduler="ddim", **kw)
    assert out.shape == (1, 4, 8, 12) and torch.isfinite(out).all()

    import diffusion.solvers as solvers

    seen = {}
    real = solvers.build_flow_s_grid

    def spy(*a, **k):
        seen["shift"] = k["shift"]
        return real(*a, **k)

    monkeypatch.setattr(solvers, "build_flow_s_grid", spy)
    out = diff.sample_loop(
        m, (1, 4, 8, 12), flow_matching_sample=True, flow_solver="euler", flow_shift_ref_hw=(8, 8), **kw
    )
    assert out.shape == (1, 4, 8, 12) and torch.isfinite(out).all()
    assert seen["shift"] == pytest.approx(math.sqrt(96 / 64))


@pytest.mark.parametrize("flow", [False, True])
def test_hires_renoise_and_refine_on_a_larger_grid(flow):
    diff = create_diffusion()
    x0 = torch.randn(1, 4, 8, 12)
    _, t_native = renoise_for_grid(diff, x0, 8, 12, 0.35, native_latent_hw=(8, 12), flow=flow)
    x_t, t_big = renoise_for_grid(diff, x0, 16, 24, 0.35, native_latent_hw=(8, 12), flow=flow)
    assert x_t.shape == (1, 4, 16, 24)
    assert t_big > t_native  # 4x the tokens: SD3 shift 2 starts the refine at more noise
    m = _tiny()
    txt = torch.randn(1, 5, 16)
    out = diff.sample_loop(
        m,
        x_t.shape,
        model_kwargs_cond={"encoder_hidden_states": txt},
        model_kwargs_uncond={"encoder_hidden_states": torch.zeros_like(txt)},
        cfg_scale=2.0,
        num_inference_steps=3,
        device="cpu",
        x_init=x_t,
        start_timestep=t_big,
        scheduler="euler",
        solver="ddim",
        flow_matching_sample=flow,
        flow_solver="euler",
        flow_shift_ref_hw=(8, 12),
    )
    assert out.shape == (1, 4, 16, 24) and torch.isfinite(out).all()
