"""Composable guidance stack: paper formulas, per-sample independence, PAG hook, sampler wiring."""

from __future__ import annotations

import pytest
import torch
from diffusion import create_diffusion
from models.attention import SelfAttention
from models.dit_text import DiT_Text
from utils.generation.cfg_batched import batched_cfg_forward
from utils.generation.guidance_stack import (
    GuidanceConfig,
    GuidanceState,
    cfg_zero_star_scale,
    combine_guided_prediction,
    laplacian_split,
    parse_sigma_interval,
    project,
    rescale_guided,
    tcfg_uncond,
)

ALL_ON = GuidanceConfig(
    rescale=0.7, apg_eta=0.0, apg_norm_threshold=3.0, fdg_low=0.5, fdg_high=1.5, tcfg_damping=1.0, cfg_zero_star=True
)


def _pair(b: int = 2, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(b, 4, 8, 8, generator=g) for _ in range(3))  # type: ignore[return-value]


def _eps_affine(abar: float) -> tuple[float, float]:
    return 1.0 / abar**0.5, -((1.0 - abar) ** 0.5) / abar**0.5


def test_plain_cfg_is_exact_in_any_space() -> None:
    c, u, x = _pair()
    out = combine_guided_prediction(c, u, x, cfg_scale=6.0, x0_affine=_eps_affine(0.3))
    assert torch.allclose(out, u + 6.0 * (c - u), atol=1e-5)


def test_every_technique_is_per_sample() -> None:
    c, u, x = _pair(b=3)
    aff = _eps_affine(0.5)
    batched = combine_guided_prediction(c, u, x, cfg_scale=5.0, config=ALL_ON, x0_affine=aff)
    for i in range(3):
        s = slice(i, i + 1)
        one = combine_guided_prediction(c[s], u[s], x[s], cfg_scale=5.0, config=ALL_ON, x0_affine=aff)
        assert torch.allclose(batched[s], one, atol=1e-5)


def test_rescale_matches_lin_formula() -> None:
    g, c, _ = _pair()
    out = rescale_guided(g, c, 0.7)
    for i in range(2):
        ref = 0.7 * g[i] * (c[i].std() / g[i].std()) + 0.3 * g[i]
        assert torch.allclose(out[i], ref, atol=1e-5)
    assert torch.allclose(rescale_guided(g, c, 1.0).std(dim=(1, 2, 3)), c.std(dim=(1, 2, 3)), atol=1e-5)


def test_project_is_orthogonal_per_sample() -> None:
    v, ref, _ = _pair()
    par, orth = project(v, ref)
    assert torch.allclose(par + orth, v, atol=1e-6)
    assert (orth * ref).sum(dim=(1, 2, 3)).abs().max() < 1e-4
    cos = torch.nn.functional.cosine_similarity(par.flatten(1), ref.flatten(1))
    assert torch.allclose(cos.abs(), torch.ones(2), atol=1e-5)


def test_apg_eta0_update_is_orthogonal_to_cond_x0() -> None:
    c, u, x = _pair()
    a, b = _eps_affine(0.4)
    out = combine_guided_prediction(c, u, x, cfg_scale=7.0, config=GuidanceConfig(apg_eta=0.0), x0_affine=(a, b))
    d_cond, d_out = a * x + b * c, a * x + b * out
    assert ((d_out - d_cond) * d_cond).sum(dim=(1, 2, 3)).abs().max() < 1e-3
    assert torch.allclose(
        combine_guided_prediction(c, u, x, cfg_scale=7.0, config=GuidanceConfig(apg_eta=1.0), x0_affine=(a, b)),
        u + 7.0 * (c - u),
        atol=1e-4,
    )


def test_apg_reverse_momentum_accumulates_before_projection() -> None:
    c1, u1, x = _pair(seed=1)
    c2, u2, _ = _pair(seed=2)
    cfg = GuidanceConfig(apg_eta=0.0, apg_momentum=-0.5)
    state = GuidanceState()
    combine_guided_prediction(c1, u1, x, cfg_scale=4.0, config=cfg, state=state)
    combine_guided_prediction(c2, u2, x, cfg_scale=4.0, config=cfg, state=state)
    assert torch.allclose(state.apg_momentum, (c2 - u2) - 0.5 * (c1 - u1), atol=1e-6)


def test_apg_norm_threshold_caps_each_sample() -> None:
    c, u, x = _pair()
    u = u.clone()
    u[0] = c[0] - 1e-3 * torch.randn_like(c[0])  # tiny delta stays uncapped
    cfg = GuidanceConfig(apg_eta=1.0, apg_norm_threshold=1.0)
    out = combine_guided_prediction(c, u, x, cfg_scale=3.0, config=cfg)
    upd = (out - c) / 2.0
    assert torch.allclose(upd[0], c[0] - u[0], atol=1e-6)
    assert abs(float(upd[1].norm()) - 1.0) < 1e-4


def test_laplacian_split_reconstructs_and_weights_bands() -> None:
    _, _, x = _pair()
    low, high = laplacian_split(x)
    assert torch.allclose(low + high, x)
    smooth = torch.ones(1, 4, 8, 8) * 0.3
    assert laplacian_split(smooth)[1].abs().max() < 1e-6
    c = torch.randn(1, 4, 8, 8)
    out = combine_guided_prediction(c, c - smooth, c, cfg_scale=5.0, config=GuidanceConfig(fdg_low=0.5, fdg_high=2.0))
    assert torch.allclose(out - c, (5.0 - 1.0) * 0.5 * smooth, atol=1e-5)
    c, u, x = _pair()
    lo, hi = laplacian_split(c - u)
    out = combine_guided_prediction(c, u, x, cfg_scale=5.0, config=GuidanceConfig(fdg_low=0.5, fdg_high=2.0))
    assert torch.allclose(out, c + 4.0 * (0.5 * lo + 2.0 * hi), atol=1e-5)


def test_cfg_zero_star_scale_and_flow_only_zero_init() -> None:
    u = torch.randn(2, 4, 4, 4)
    assert torch.allclose(cfg_zero_star_scale(2.0 * u, u).flatten(), torch.full((2,), 2.0), atol=1e-5)
    c = torch.randn(2, 4, 4, 4)
    cfg = GuidanceConfig(cfg_zero_star=True)
    s = cfg_zero_star_scale(c, u)
    out = combine_guided_prediction(c, u, c, cfg_scale=4.0, config=cfg)
    assert torch.allclose(out, s * u + 4.0 * (c - s * u), atol=1e-5)
    assert cfg.zero_init_active(0, 50, flow=True)
    assert not cfg.zero_init_active(0, 50, flow=False)
    assert not GuidanceConfig(cfg_zero_star=True, cfg_zero_init_frac=0.0).zero_init_active(0, 50, flow=True)


def test_tcfg_full_damping_keeps_uncond_on_principal_direction() -> None:
    c, u, _ = _pair()
    assert torch.allclose(tcfg_uncond(c, u, damping=0.0), u, atol=1e-4)
    damped = tcfg_uncond(c, u, damping=1.0)
    m = torch.stack([c, u], dim=1).flatten(2)
    v1 = torch.linalg.svd(m, full_matrices=False)[2][:, 0]
    cos = torch.nn.functional.cosine_similarity(damped.flatten(1), v1)
    assert torch.allclose(cos.abs(), torch.ones(2), atol=1e-4)


def test_interval_parsing_and_gate() -> None:
    assert parse_sigma_interval("0.28,5.42") == (0.28, 5.42)
    assert parse_sigma_interval("") is None
    cfg = GuidanceConfig(interval=(0.28, 5.42))
    assert cfg.cfg_active(1.0) and not cfg.cfg_active(10.0) and not cfg.cfg_active(0.28)
    with pytest.raises(ValueError):
        parse_sigma_interval("5,1")


class _CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[int] = []

    def forward(self, x, t, encoder_hidden_states=None, **kw):
        self.batches.append(x.shape[0])
        return x * 0.5 + encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1)


def test_interval_skips_uncond_forward_and_composes_with_rescale() -> None:
    model = _CountingModel()
    x = torch.randn(2, 4, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    mk_c = {"encoder_hidden_states": torch.randn(2, 3, 5)}
    mk_u = {"encoder_hidden_states": torch.randn(2, 3, 5)}
    cfg = GuidanceConfig(interval=(0.5, 2.0))
    kw = dict(model_kwargs_cond=mk_c, model_kwargs_uncond=mk_u, cfg_scale=5.0, guidance=cfg)
    out = batched_cfg_forward(model, x, t, sigma=10.0, **kw)
    assert model.batches == [2] and torch.allclose(out, model(x, t, **mk_c))
    model.batches.clear()
    batched_cfg_forward(model, x, t, sigma=1.0, **kw)
    assert model.batches == [4]


def test_identity_attention_returns_value_projection() -> None:
    torch.manual_seed(0)
    attn = SelfAttention(16, 4).eval()
    x = torch.randn(2, 6, 16)
    v = attn.qkv(x).reshape(2, 6, 3, 16)[:, :, 2]
    full = attn(x, use_xformers=False)
    mixed = attn(x, use_xformers=False, identity_from=1)
    assert torch.allclose(mixed[0], full[0], atol=1e-6)
    assert torch.allclose(mixed[1], attn.out_proj(v[1]), atol=1e-6)


def _tiny_dit() -> DiT_Text:
    torch.manual_seed(0)
    m = DiT_Text(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=4,
        num_heads=4,
        text_dim=16,
        learn_sigma=False,
        use_xformers=False,
    )
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return m.eval()


@torch.no_grad()
def test_pag_batched_matches_separate_forwards() -> None:
    model = _tiny_dit()
    x = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 500, dtype=torch.long)
    mk_c = {"encoder_hidden_states": torch.randn(2, 5, 16)}
    mk_u = {"encoder_hidden_states": torch.zeros(2, 5, 16)}
    cfg = GuidanceConfig(pag_scale=3.0, pag_blocks=(2,))
    out = batched_cfg_forward(
        model, x, t, model_kwargs_cond=mk_c, model_kwargs_uncond=mk_u, cfg_scale=4.0, guidance=cfg
    )
    oc, ou = model(x, t, **mk_c), model(x, t, **mk_u)
    op = model(x, t, pag_blocks=(2,), pag_from=0, **mk_c)
    assert not torch.allclose(op, oc, atol=1e-4)
    assert torch.allclose(out, ou + 4.0 * (oc - ou) + 3.0 * (oc - op), atol=1e-4)


@torch.no_grad()
@pytest.mark.parametrize("flow", [False, True])
def test_sample_loop_runs_full_stack(flow: bool) -> None:
    model = _tiny_dit()
    diff = create_diffusion(num_timesteps=100)
    mk_c = {"encoder_hidden_states": torch.randn(1, 5, 16)}
    mk_u = {"encoder_hidden_states": torch.zeros(1, 5, 16)}
    out = diff.sample_loop(
        model,
        (1, 4, 8, 8),
        model_kwargs_cond=mk_c,
        model_kwargs_uncond=mk_u,
        cfg_scale=4.0,
        num_inference_steps=4,
        device="cpu",
        flow_matching_sample=flow,
        cfg_rescale=0.5,
        apg_parallel_eta=0.0,
        apg_momentum_beta=-0.5,
        fdg_low=0.6,
        fdg_high=1.3,
        cfg_interval="0.1,50",
        tcfg_damping=1.0,
        cfg_zero_star=True,
        pag_scale=2.0,
        slg_scale=1.0,
        slg_skip_blocks="auto",
        slg_stop_frac=1.0,
    )
    assert out.shape == (1, 4, 8, 8) and torch.isfinite(out).all()
