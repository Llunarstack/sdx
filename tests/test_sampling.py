"""Tests for ``diffusion.sampling`` (holy-grail recipe) and the latent helpers in ``diffusion.sampling_utils``.

Verifies per-step plans, presets, runtime guard clamps, CADS and latent refinement helpers.

Run with:
    pytest tests/test_sampling.py -v
Also holds what used to be: test_samplers, test_sampling_flex, test_apex_sampler, test_solvers_schedules, test_cfg_batched, test_guidance_stack, test_resolution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from diffusion.gaussian_diffusion import create_diffusion, dynamic_threshold
from diffusion.sampling import sampling_canvas_hw
from diffusion.schedules import (
    flux_resolution_shift,
    inverse_time_shift,
    sd3_resolution_shift,
    time_shift,
    vp_shifted_timestep,
)
from diffusion.solvers import EDM_SIGMA_MAX, EDM_SIGMA_MIN, build_flow_s_grid, build_inference_timesteps, karras_sigmas
from models.attention import SelfAttention, create_block_causal_mask_2d
from models.dit import get_2d_sincos_pos_embed, grid_pos_embed
from models.dit_text import DiT_Predecessor_Text, DiT_Supreme_Text, DiT_Text, grid_ar_mask
from utils.generation.guidance import (
    GuidanceConfig,
    GuidanceState,
    batched_cfg_forward,
    cfg_zero_star_scale,
    combine_guided_prediction,
    laplacian_split,
    merge_cfg_model_kwargs,
    parse_sigma_interval,
    project,
    rescale_guided,
    tcfg_uncond,
)
from utils.generation.sample_core import renoise_for_grid

# ---------------------------------------------------------------------------
# Blueprint / step plan
# ---------------------------------------------------------------------------


class TestHolyGrailBlueprint:
    def test_step_plan_at_start(self):
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5)
        plan = build_holy_grail_step_plan(recipe=recipe, step_index=0, total_steps=50)
        assert plan.cfg_scale > 0
        assert plan.control_scale > 0
        assert plan.adapter_scale > 0

    def test_step_plan_at_end(self):
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5)
        plan = build_holy_grail_step_plan(recipe=recipe, step_index=49, total_steps=50)
        assert plan.cfg_scale > 0

    def test_cfg_increases_over_steps(self):
        """With cfg_early_ratio < cfg_late_ratio, CFG should increase over steps."""
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5, cfg_early_ratio=0.7, cfg_late_ratio=1.0)
        plan_start = build_holy_grail_step_plan(recipe=recipe, step_index=0, total_steps=50)
        plan_end = build_holy_grail_step_plan(recipe=recipe, step_index=49, total_steps=50)
        assert plan_end.cfg_scale > plan_start.cfg_scale

    def test_single_step(self):
        from diffusion.sampling import HolyGrailRecipe, build_holy_grail_step_plan

        recipe = HolyGrailRecipe(base_cfg=7.5)
        plan = build_holy_grail_step_plan(recipe=recipe, step_index=0, total_steps=1)
        assert plan.cfg_scale > 0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestHolyGrailPresets:
    def test_list_presets(self):
        from diffusion.sampling import list_holy_grail_presets

        presets = list_holy_grail_presets()
        assert "balanced" in presets
        assert "anime" in presets
        assert "photoreal" in presets

    def test_get_preset(self):
        from diffusion.sampling import get_holy_grail_preset

        p = get_holy_grail_preset("anime")
        assert p.name == "anime"
        assert p.cfg_early_ratio > 0

    def test_get_unknown_preset_raises(self):
        from diffusion.sampling import get_holy_grail_preset

        with pytest.raises(KeyError):
            get_holy_grail_preset("nonexistent_preset_xyz")

    def test_preset_values_sane(self):
        from diffusion.sampling import HOLY_GRAIL_PRESETS

        for name, p in HOLY_GRAIL_PRESETS.items():
            assert 0 < p.cfg_early_ratio <= 2.0, f"{name}: bad cfg_early_ratio"
            assert 0 < p.cfg_late_ratio <= 2.0, f"{name}: bad cfg_late_ratio"
            assert p.cads_strength >= 0, f"{name}: negative cads_strength"
            assert 0 < p.clamp_floor, f"{name}: non-positive clamp_floor"


# ---------------------------------------------------------------------------
# Runtime guard
# ---------------------------------------------------------------------------


class TestRuntimeGuard:
    def test_sanitize_clamps_cfg_ratios(self):
        from diffusion.sampling import sanitize_holy_grail_kwargs

        out = sanitize_holy_grail_kwargs({"holy_grail_cfg_early_ratio": 99.0})
        assert out["holy_grail_cfg_early_ratio"] <= 1.4

    def test_sanitize_clamps_negative(self):
        from diffusion.sampling import sanitize_holy_grail_kwargs

        out = sanitize_holy_grail_kwargs({"holy_grail_cads_strength": -1.0})
        assert out["holy_grail_cads_strength"] == 0.0

    def test_sanitize_consistency_constraint(self):
        """cads_min_strength must not exceed cads_strength after sanitize."""
        from diffusion.sampling import sanitize_holy_grail_kwargs

        out = sanitize_holy_grail_kwargs(
            {
                "holy_grail_cads_strength": 0.05,
                "holy_grail_cads_min_strength": 0.1,  # violates constraint
            }
        )
        assert out["holy_grail_cads_min_strength"] <= out["holy_grail_cads_strength"]

    def test_sanitize_returns_copy(self):
        from diffusion.sampling import sanitize_holy_grail_kwargs

        original = {"holy_grail_cfg_early_ratio": 0.72}
        out = sanitize_holy_grail_kwargs(original)
        assert out is not original


# ---------------------------------------------------------------------------
# Condition annealing
# ---------------------------------------------------------------------------


class TestConditionAnnealing:
    def test_cads_noise_std_at_start(self):
        from diffusion.sampling import cads_noise_std

        std = cads_noise_std(progress=0.0, base_strength=0.1)
        assert abs(std - 0.1) < 1e-6

    def test_cads_noise_std_at_end(self):
        from diffusion.sampling import cads_noise_std

        std = cads_noise_std(progress=1.0, base_strength=0.1, min_strength=0.0)
        assert std == 0.0

    def test_apply_condition_noise_shape(self):
        from diffusion.sampling import apply_condition_noise

        cond = torch.randn(2, 77, 4096)
        noisy = apply_condition_noise(cond, std=0.05)
        assert noisy.shape == cond.shape

    def test_apply_condition_noise_zero_std_noop(self):
        from diffusion.sampling import apply_condition_noise

        cond = torch.randn(2, 77, 4096)
        out = apply_condition_noise(cond, std=0.0)
        torch.testing.assert_close(out, cond)


# ---------------------------------------------------------------------------
# Latent refiner
# ---------------------------------------------------------------------------


class TestLatentRefiner:
    def test_unsharp_mask_shape(self):
        from diffusion.gaussian_diffusion import unsharp_mask_latent

        x = torch.randn(2, 4, 16, 16)
        out = unsharp_mask_latent(x, sigma=0.6, amount=0.2)
        assert out.shape == x.shape

    def test_unsharp_mask_zero_amount_noop(self):
        from diffusion.gaussian_diffusion import unsharp_mask_latent

        x = torch.randn(2, 4, 16, 16)
        out = unsharp_mask_latent(x, sigma=0.6, amount=0.0)
        torch.testing.assert_close(out, x)

    def test_dynamic_threshold_clip_only_keeps_inliers(self):
        """``rescale=False`` (holy-grail final clamp) clips outliers but never shrinks the latent."""
        from diffusion.gaussian_diffusion import dynamic_threshold

        x = torch.randn(2, 4, 16, 16) * 3
        out = dynamic_threshold(x, 0.99, 1.0, rescale=False)
        bound = torch.quantile(x.flatten(1).abs(), 0.99, dim=1).view(2, 1, 1, 1)
        inside = x.abs() <= bound
        torch.testing.assert_close(out[inside], x[inside])
        assert (out.abs() <= bound + 1e-6).all()


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


class TestRecommender:
    def test_anime_prompt(self):
        from diffusion.sampling import recommend_holy_grail_preset

        assert recommend_holy_grail_preset(prompt="anime girl, manga style") == "anime"

    def test_photoreal_prompt(self):
        from diffusion.sampling import recommend_holy_grail_preset

        assert recommend_holy_grail_preset(prompt="photoreal portrait, dslr") == "photoreal"

    def test_illustration_prompt(self):
        from diffusion.sampling import recommend_holy_grail_preset

        assert recommend_holy_grail_preset(prompt="concept art illustration") == "illustration"

    def test_default_balanced(self):
        from diffusion.sampling import recommend_holy_grail_preset

        result = recommend_holy_grail_preset(prompt="a dog in a field")
        assert result in ("balanced", "photoreal", "anime", "illustration", "aggressive")


# ==================================================================================================
# test_samplers: Sampler math: Karras grids, solvers on an exact Gaussian denoiser, SDE noise, Restart, thresholding.
# ==================================================================================================


MEAN, STD = 0.5, 0.7
SHAPE = (4096, 1, 1, 1)


class GaussEps(torch.nn.Module):
    """Exact noise prediction for VP data x0 ~ N(MEAN, STD^2); ``bias_above`` inflates it at higher t."""

    def __init__(self, diffusion, bias_above: int | None = None, bias: float = 1.0):
        super().__init__()
        self.abar, self.bias_above, self.bias = diffusion.alpha_cumprod, bias_above, bias

    def forward(self, x, t, **_):
        ab = self.abar.to(x.device)[t].view(-1, 1, 1, 1)
        a, sg = ab.sqrt(), (1 - ab).sqrt()
        x0 = MEAN + a * STD**2 / (ab * STD**2 + sg**2) * (x - a * MEAN)
        eps = (x - a * x0) / sg
        if self.bias_above is not None:
            eps = torch.where(t.view(-1, 1, 1, 1) > self.bias_above, eps * self.bias, eps)
        return eps


class GaussVelocity(torch.nn.Module):
    """Exact rectified-flow velocity for x0 ~ N(MEAN, STD^2); records the timesteps it is called at."""

    def __init__(self, num_timesteps: int = 1000):
        super().__init__()
        self.T, self.calls = num_timesteps, []

    def forward(self, x, t, **_):
        self.calls.append(int(t[0]))
        s = (t.float() / (self.T - 1)).view(-1, 1, 1, 1)
        a = 1 - s
        x0 = MEAN + a * STD**2 / (a * a * STD**2 + s * s) * (x - a * MEAN)
        return (x - a * x0) / s.clamp_min(1e-8) - x0


def _sample(diff, model, seed=0, **kw):
    torch.manual_seed(seed)
    return diff.sample_loop(model, SHAPE, cfg_scale=1.0, device="cpu", **kw)


def _ref_std(seed=0):
    torch.manual_seed(seed)
    return STD * torch.randn(SHAPE).std().item()


def test_karras_sigmas_match_edm_formula():
    sig = karras_sigmas(80.0, 0.002, 10, rho=7.0)
    i = np.arange(10)
    edm = (80.0 ** (1 / 7) + i / 9 * (0.002 ** (1 / 7) - 80.0 ** (1 / 7))) ** 7
    np.testing.assert_allclose(sig, edm, rtol=1e-12)
    assert sig[0] == pytest.approx(80.0) and sig[-1] == pytest.approx(0.002)
    assert np.all(np.diff(sig) < 0)


@pytest.mark.parametrize("n", [8, 20, 50])
def test_vp_karras_rho_is_monotone_karras(n):
    ac = create_diffusion().alpha_cumprod.numpy().astype(np.float64)
    idx = build_inference_timesteps("karras_rho", 1000, n, ac)
    assert len(idx) == n and idx[0] == 999 and idx[-1] == 0
    assert np.all(np.diff(idx) < 0)
    sig = np.sqrt((1 - ac[idx]) / ac[idx])
    target = karras_sigmas(sig[0], sig[-1], n)
    mid = slice(1, n // 2)  # high-noise half: index snapping is fine there
    np.testing.assert_allclose(sig[mid], target[mid], rtol=0.05)


@pytest.mark.parametrize("n", [8, 20])
def test_flow_karras_grid_has_no_zero_length_steps(n):
    g = build_flow_s_grid("karras", n)
    assert g.shape == (n + 1,) and g[0] == 1.0 and g[-1] == 0.0
    assert np.all(-np.diff(g) > 1e-3)
    sig = karras_sigmas(EDM_SIGMA_MAX, EDM_SIGMA_MIN, n)
    np.testing.assert_allclose(g[1:-1], (sig / (1 + sig))[1:], rtol=1e-12)


@pytest.mark.parametrize("solver", ["ddim", "heun", "dpmpp_2m", "dpmpp_3m", "unipc", "apex"])
def test_vp_solvers_converge_on_gaussian(solver):
    diff = create_diffusion()
    out = _sample(diff, GaussEps(diff), num_inference_steps=24, solver=solver, scheduler="snr_uniform")
    tol = 0.08 if solver == "ddim" else 0.03
    assert out.mean().item() == pytest.approx(MEAN, abs=0.02)
    assert out.std().item() == pytest.approx(_ref_std(), abs=tol)


@pytest.mark.parametrize("solver", ["euler", "heun", "midpoint", "dpmpp_2m", "apex"])
def test_flow_solvers_converge_on_gaussian(solver):
    out = _sample(
        create_diffusion(), GaussVelocity(), num_inference_steps=20, flow_matching_sample=True, flow_solver=solver
    )
    assert out.std().item() == pytest.approx(_ref_std(), abs=0.05 if solver == "euler" else 0.01)


def test_flow_heun_takes_euler_last_step():
    model = GaussVelocity()
    n = 6
    _sample(create_diffusion(), model, num_inference_steps=n, flow_matching_sample=True, flow_solver="heun")
    assert len(model.calls) == 2 * n - 1
    assert min(model.calls) > 0


@pytest.mark.parametrize("flow", [False, True])
def test_sde_solver_is_seeded_and_reduces_to_ode(flow):
    diff = create_diffusion()
    model = GaussVelocity() if flow else GaussEps(diff)
    kw = dict(flow_matching_sample=True, flow_schedule="ays") if flow else dict(scheduler="snr_uniform", karras_rho=7.0)
    key = "flow_solver" if flow else "solver"
    a = _sample(diff, model, seed=1, num_inference_steps=10, **{key: "dpmpp_2m_sde"}, **kw)
    b = _sample(diff, model, seed=1, num_inference_steps=10, **{key: "dpmpp_2m_sde"}, **kw)
    c = _sample(diff, model, seed=2, num_inference_steps=10, **{key: "dpmpp_2m_sde"}, **kw)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(a, c)
    ode = _sample(diff, model, seed=1, num_inference_steps=10, **{key: "dpmpp_2m"}, **kw)
    torch.testing.assert_close(
        _sample(diff, model, seed=1, num_inference_steps=10, eta=0.0, **{key: "dpmpp_2m_sde"}, **kw), ode
    )
    off_band = _sample(diff, model, seed=1, num_inference_steps=10, sde_band="1e6,1e7", **{key: "dpmpp_2m_sde"}, **kw)
    torch.testing.assert_close(off_band, ode)


def test_sde_preserves_the_data_distribution():
    diff = create_diffusion()
    out = _sample(diff, GaussEps(diff), num_inference_steps=40, solver="dpmpp_2m_sde", scheduler="snr_uniform")
    assert out.mean().item() == pytest.approx(MEAN, abs=0.03)
    assert out.std().item() == pytest.approx(STD, abs=0.03)


def test_noise_options_reject_ode_solvers():
    diff = create_diffusion()
    with pytest.raises(ValueError, match="stochastic solver"):
        _sample(diff, GaussEps(diff), num_inference_steps=3, solver="dpmpp_2m", eta=0.5)
    with pytest.raises(ValueError, match="step_noise_scales"):
        _sample(diff, GaussEps(diff), num_inference_steps=3, solver="unipc", step_noise_scales=[1.0])
    with pytest.raises(ValueError, match="stochastic solver"):
        _sample(diff, GaussVelocity(), num_inference_steps=3, flow_matching_sample=True, flow_solver="euler", eta=1.0)


def test_restart_contracts_error_from_a_biased_denoiser():
    """Restart (Xu et al. 2023): noise added up to t_max forgets errors made above t_max."""
    diff = create_diffusion()
    model = GaussEps(diff, bias_above=300, bias=1.25)
    x0 = _sample(diff, model, num_inference_steps=30, solver="dpmpp_2m", scheduler="snr_uniform")

    def solve(x_t, t):
        return diff.sample_loop(
            model,
            SHAPE,
            cfg_scale=1.0,
            device="cpu",
            x_init=x_t,
            start_timestep=t,
            num_inference_steps=30,
            solver="dpmpp_2m",
        )

    err_before = abs(x0.std().item() - STD)
    torch.manual_seed(3)
    err_after = abs(diff.restart(x0, solve, t_max=300, cycles=3).std().item() - STD)
    assert err_before > 0.05
    assert err_after < 0.5 * err_before


def test_flow_resume_uses_x_init_as_the_noised_state():
    class Zero(torch.nn.Module):
        def forward(self, x, t, **_):
            return torch.zeros_like(x)

    x_t = torch.randn(2, 4, 4, 4)
    out = create_diffusion().sample_loop(
        Zero(),
        x_t.shape,
        cfg_scale=1.0,
        device="cpu",
        x_init=x_t,
        start_timestep=500,
        num_inference_steps=4,
        flow_matching_sample=True,
        flow_solver="euler",
    )
    torch.testing.assert_close(out, x_t)


def test_eps_scaling_keeps_more_noise():
    diff = create_diffusion()
    base = _sample(diff, GaussEps(diff), num_inference_steps=20, solver="ddim", scheduler="snr_uniform")
    same = _sample(diff, GaussEps(diff), num_inference_steps=20, solver="ddim", scheduler="snr_uniform", eps_scale=1.0)
    scaled = _sample(
        diff, GaussEps(diff), num_inference_steps=20, solver="ddim", scheduler="snr_uniform", eps_scale=1.01
    )
    torch.testing.assert_close(base, same, rtol=0, atol=0)
    assert scaled.std() > base.std()


def test_dynamic_threshold_matches_numpy_reference_and_imagen():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 8, 8) * torch.tensor([0.2, 2.0, 4.0]).view(3, 1, 1, 1)
    flat = x.reshape(3, -1).numpy()
    bound = np.maximum(np.quantile(np.abs(flat), 0.995, axis=1, keepdims=True), 1.0)
    ref = (np.clip(flat, -bound, bound) / bound).reshape(x.shape)
    ours = dynamic_threshold(x, 0.995, 1.0)
    np.testing.assert_allclose(ours.numpy(), ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ours[0], x[0])
    assert ours.abs().max() <= 1.0 + 1e-6
    torch.testing.assert_close(create_diffusion()._dynamic_threshold(x, percentile=99.5), ours)
    inside = torch.rand(2, 4, 8, 8) * 3.9
    torch.testing.assert_close(dynamic_threshold(inside, 0.995, 4.0), inside)


def test_sampler_accepts_sanitized_feature_kwargs():
    from frontier.layout import AttentionLayoutPlan
    from utils.generation.guidance import sanitize_diffusion_feature_kwargs

    plan = AttentionLayoutPlan(region_names=("a",), boxes=((0.0, 0.0, 0.5, 1.0),), enforce_steps=(0, 1))
    feats = sanitize_diffusion_feature_kwargs(
        {"attention_layout_plan": plan, "step_emphasis": [2.0, 1.0], "frontier_compute_cost": 3, "cads_mix_ratio": 0.2}
    )
    assert set(feats) == {"attention_layout_plan", "step_noise_scales"}
    diff = create_diffusion()
    out = diff.sample_loop(
        GaussEps(diff), (1, 4, 4, 4), cfg_scale=1.0, device="cpu", num_inference_steps=3, solver="dpmpp_2m_sde", **feats
    )
    assert torch.isfinite(out).all()


# ==================================================================================================
# test_sampling_flex: Sampler / timestep / CFG schedule composition (torch-free where possible).
# ==================================================================================================


def test_vp_solver_aliases_resolve() -> None:
    from diffusion.gaussian_diffusion import canonicalize_vp_solver

    assert canonicalize_vp_solver("ddim") == "ddim"
    assert canonicalize_vp_solver("euler") == "ddim"
    assert canonicalize_vp_solver("RK2") == "heun"


def test_flow_solver_aliases_resolve() -> None:
    from diffusion.gaussian_diffusion import canonicalize_flow_solver

    assert canonicalize_flow_solver("Euler") == "euler"
    assert canonicalize_flow_solver("edm-heun") == "heun"


def test_inference_timesteps_indices_prefix() -> None:
    from diffusion.solvers import build_inference_timesteps

    ac = np.clip(np.linspace(0.999, 0.01, 200), 1e-4, 1.0 - 1e-4)
    out = build_inference_timesteps("indices:199,160,120,80,40,10,0", 200, 7, ac)
    assert out.shape == (7,)
    assert out[0] >= out[-1]


def test_snr_cfg_multiplier_bounded() -> None:
    from diffusion.gaussian_diffusion import cfg_scale_snr_aware_multiplier

    a_hi = cfg_scale_snr_aware_multiplier(0.995)
    a_lo = cfg_scale_snr_aware_multiplier(0.02)
    assert 0 < a_hi < 10 and 0 < a_lo < 10


# ==================================================================================================
# test_apex_sampler: APEX sampler/scheduler unit tests.
# ==================================================================================================


class TestApexSchedule:
    def test_vp_indices_descending(self):
        from diffusion.schedules import get_beta_schedule
        from diffusion.solvers import build_inference_timesteps, list_timestep_schedules

        assert "apex" in list_timestep_schedules()
        beta = get_beta_schedule("linear", 1000)
        ac = np.cumprod(1.0 - beta)
        idx = build_inference_timesteps("apex", 1000, 28, ac, karras_rho=7.0)
        assert idx.shape[0] == 28
        assert np.all(np.diff(idx.astype(np.int64)) < 0)
        assert int(idx[-1]) == 0

    def test_flow_grid(self):
        from diffusion.solvers import build_flow_s_grid, list_flow_schedules

        assert "apex" in list_flow_schedules()
        g = build_flow_s_grid("apex", 20)
        assert g.shape == (21,)
        assert abs(g[0] - 1.0) < 1e-9
        assert abs(g[-1] - 0.0) < 1e-9
        assert np.all(np.diff(g) <= 1e-12)


class TestApexSolver:
    def test_aliases(self):
        from diffusion.gaussian_diffusion import canonicalize_flow_solver, canonicalize_vp_solver

        assert canonicalize_vp_solver("apex") == "apex"
        assert canonicalize_vp_solver("apex_adaptive") == "apex_adaptive"
        assert canonicalize_flow_solver("apex") == "apex"

    def test_vp_step_reduces_noise_with_perfect_x0(self):
        from diffusion.solvers import ApexController, SolverState, vp_apex_update

        B, C, H, W = 1, 4, 8, 8
        x0 = torch.randn(B, C, H, W)
        ab_cur = torch.full((B,), 0.25)
        ab_next = torch.full((B,), 0.65)
        noise = torch.randn_like(x0)
        x = ab_cur.sqrt().view(B, 1, 1, 1) * x0 + (1 - ab_cur).sqrt().view(B, 1, 1, 1) * noise
        ctrl = ApexController(atol=0.05, rtol=0.1)
        state = SolverState(max_order=3)
        res = vp_apex_update(
            x=x,
            x0_pred=x0,
            alpha_bar_cur=ab_cur,
            alpha_bar_next=ab_next,
            state=state,
            controller=ctrl,
            progress=0.4,
        )
        assert res.x_next.shape == x.shape
        assert res.error >= 0.0
        assert ctrl.accepted >= 1
        # Closer to clean prediction in L2 vs pure noise (heuristic)
        assert float((res.x_next - x0).pow(2).mean()) < float((x - x0).pow(2).mean()) + 1e-5

    def test_redistribute_remaining(self):
        from diffusion.schedules import get_beta_schedule
        from diffusion.solvers import redistribute_remaining_vp_indices

        beta = get_beta_schedule("linear", 1000)
        ac = np.cumprod(1.0 - beta)
        tail = redistribute_remaining_vp_indices(800, 10, 1000, ac, difficulty=2.0)
        assert len(tail) == 10
        assert int(tail[-1]) == 0
        assert np.all(np.diff(tail.astype(np.int64)) < 0)

    def test_phase_weight_peaks_mid(self):
        from diffusion.solvers import phase_weight

        mid = phase_weight(0.42)
        early = phase_weight(0.05)
        late = phase_weight(0.95)
        assert mid > early
        assert mid > late


# ==================================================================================================
# test_solvers_schedules: Tests for modern inference solvers and timestep schedules.
# ==================================================================================================


class TestSolverAliases:
    def test_dpmpp_aliases_are_real(self):
        from diffusion.gaussian_diffusion import canonicalize_vp_solver

        assert canonicalize_vp_solver("dpmpp_2m") == "dpmpp_2m"
        assert canonicalize_vp_solver("dpm++_2m") == "dpmpp_2m"
        assert canonicalize_vp_solver("dpmsolver_pp_1") == "dpmpp_2m"
        assert canonicalize_vp_solver("dpmpp_3m") == "dpmpp_3m"
        assert canonicalize_vp_solver("unipc") == "unipc"
        assert canonicalize_vp_solver("ddim") == "ddim"
        assert canonicalize_vp_solver("heun") == "heun"

    def test_flow_aliases(self):
        from diffusion.gaussian_diffusion import canonicalize_flow_solver

        assert canonicalize_flow_solver("midpoint") == "midpoint"
        assert canonicalize_flow_solver("dpmpp_2m") == "dpmpp_2m"
        assert canonicalize_flow_solver("heun") == "heun"
        assert canonicalize_flow_solver("euler") == "euler"


class TestAysSchedules:
    def test_ays_descending(self):
        from diffusion.schedules import get_beta_schedule
        from diffusion.solvers import build_inference_timesteps

        beta = get_beta_schedule("linear", 1000)
        ac = np.cumprod(1.0 - beta)
        for name in ("ays", "ays_dit"):
            idx = build_inference_timesteps(name, 1000, 28, ac)
            assert idx.shape[0] == 28
            assert np.all(np.diff(idx.astype(np.int64)) < 0), f"{name} must be strictly descending"
            assert int(idx[0]) > int(idx[-1])

    def test_ays_dit_registered(self):
        from diffusion.solvers import list_timestep_schedules

        names = set(list_timestep_schedules())
        assert "ays" in names
        assert "ays_dit" in names


class TestFlowSGrid:
    def test_linear_endpoints(self):
        from diffusion.solvers import build_flow_s_grid

        g = build_flow_s_grid("linear", 10)
        assert g.shape == (11,)
        assert abs(g[0] - 1.0) < 1e-9
        assert abs(g[-1] - 0.0) < 1e-9

    def test_ays_monotonic_decreasing(self):
        from diffusion.solvers import build_flow_s_grid

        for name in ("ays", "ays_dit", "karras"):
            g = build_flow_s_grid(name, 20)
            assert np.all(np.diff(g) <= 1e-12), f"{name} should be non-increasing in s"


class TestDpmppVpStep:
    def test_first_order_moves_toward_x0(self):
        """With a perfect x0 prediction, one DPM++ step should reduce noise."""
        from diffusion.solvers import SolverState, vp_dpmpp_update

        B, C, H, W = 2, 4, 8, 8
        x0 = torch.randn(B, C, H, W)
        # Noisy sample at mid ᾱ
        ab_cur = torch.full((B,), 0.3)
        ab_next = torch.full((B,), 0.7)
        noise = torch.randn_like(x0)
        x = ab_cur.sqrt().view(B, 1, 1, 1) * x0 + (1 - ab_cur).sqrt().view(B, 1, 1, 1) * noise
        state = SolverState(max_order=3)
        x_next, state = vp_dpmpp_update(
            x=x,
            x0_pred=x0,
            alpha_bar_cur=ab_cur,
            alpha_bar_next=ab_next,
            state=state,
            order=2,
        )
        assert x_next.shape == x.shape
        assert state.order >= 1
        # Closer to x0 in L2 than the starting noisy x (usually true for perfect x0).
        err0 = (x - x0).pow(2).mean().item()
        err1 = (x_next - x0).pow(2).mean().item()
        assert err1 < err0

    def test_second_step_uses_history(self):
        from diffusion.solvers import SolverState, vp_dpmpp_update

        B, C, H, W = 1, 3, 4, 4
        x = torch.randn(B, C, H, W)
        x0 = torch.randn_like(x)
        state = SolverState(max_order=3)
        ab_vals = [0.2, 0.4, 0.6, 0.85]
        for i in range(3):
            x, state = vp_dpmpp_update(
                x=x,
                x0_pred=x0,
                alpha_bar_cur=torch.tensor([ab_vals[i]]),
                alpha_bar_next=torch.tensor([ab_vals[i + 1]]),
                state=state,
                order=2,
            )
        assert state.order == 3


class TestFlowDpmpp:
    def test_first_step_is_euler_last_step_is_data_prediction(self):
        from diffusion.solvers import SolverState, flow_dpmpp_2m_update

        x = torch.zeros(1, 2, 2, 2)
        v = torch.ones_like(x)
        state = SolverState(max_order=3)
        x1, state = flow_dpmpp_2m_update(x=x, velocity=v, s_cur=1.0, s_next=0.5, state=state)
        # Euler: x += v * (-0.5) → -0.5
        assert torch.allclose(x1, torch.full_like(x, -0.5))
        x2, state = flow_dpmpp_2m_update(x=x1, velocity=v, s_cur=0.5, s_next=0.0, state=state)
        assert x2.shape == x.shape
        assert state.order == 2


class TestUnipc:
    def test_corrector_is_trapezoid_at_order_one(self):
        """UniC-1 averages the data predictions at both ends of the step (bh2)."""
        from diffusion.solvers import SolverState, vp_dpmpp_update, vp_unipc_correct

        x = torch.randn(1, 2, 4, 4)
        m0, m1 = torch.randn_like(x), torch.randn_like(x)
        state = SolverState(max_order=3)
        _, state = vp_dpmpp_update(x=x, x0_pred=m0, alpha_bar_cur=0.25, alpha_bar_next=0.55, state=state, order=1)
        corrected = vp_unipc_correct(x_prev=x, x0_new=m1, alpha_bar_prev=0.25, alpha_bar_cur=0.55, state=state, order=1)
        state_avg = SolverState(max_order=3)
        expected, _ = vp_dpmpp_update(
            x=x, x0_pred=0.5 * (m0 + m1), alpha_bar_cur=0.25, alpha_bar_next=0.55, state=state_avg, order=1
        )
        torch.testing.assert_close(corrected, expected)


# ==================================================================================================
# test_cfg_batched: Batched CFG helper (shape/math smoke).
# ==================================================================================================


class _TinyCfgModel(torch.nn.Module):
    def forward(self, x, t, encoder_hidden_states=None, **kwargs):
        del t, kwargs
        bias = encoder_hidden_states.mean(dim=(1, 2)).view(x.shape[0], 1, 1, 1)
        return x + bias.expand_as(x)


def test_batched_cfg_matches_two_forwards():
    model = _TinyCfgModel()
    x = torch.randn(2, 4, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    emb_c = torch.randn(2, 3, 16)
    emb_u = torch.randn(2, 3, 16)
    mk_c = {"encoder_hidden_states": emb_c}
    mk_u = {"encoder_hidden_states": emb_u}
    batched = batched_cfg_forward(model, x, t, model_kwargs_cond=mk_c, model_kwargs_uncond=mk_u, cfg_scale=7.5)
    oc = model(x, t, **mk_c)
    ou = model(x, t, **mk_u)
    sequential = ou + 7.5 * (oc - ou)
    assert torch.allclose(batched, sequential, atol=1e-5)


def test_merge_cfg_kwargs_doubles_batch():
    mk_c = {"encoder_hidden_states": torch.randn(2, 3, 4)}
    mk_u = {"encoder_hidden_states": torch.randn(2, 3, 4)}
    merged = merge_cfg_model_kwargs(mk_c, mk_u)
    assert merged["encoder_hidden_states"].shape[0] == 4


# ==================================================================================================
# test_guidance_stack: Composable guidance stack: paper formulas, per-sample independence, PAG hook, sampler wiring.
# ==================================================================================================


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


# ==================================================================================================
# test_resolution: Resolution / aspect-ratio correctness: any-grid DiTs, old checkpoints, SD3 shift, canvas, rectangular sampling.
# ==================================================================================================


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
