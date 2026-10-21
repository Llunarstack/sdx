"""Sampler math: Karras grids, solvers on an exact Gaussian denoiser, SDE noise, Restart, thresholding."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusion.gaussian_diffusion import create_diffusion
from diffusion.inference_timesteps import EDM_SIGMA_MAX, EDM_SIGMA_MIN, build_inference_timesteps, karras_sigmas
from diffusion.sampling_utils import dynamic_threshold
from diffusion.solvers import build_flow_s_grid

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
    from frontier.attention import AttentionLayoutPlan
    from utils.generation.frontier_consume import sanitize_diffusion_feature_kwargs

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
