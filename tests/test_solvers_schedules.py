"""Tests for modern inference solvers and timestep schedules."""

from __future__ import annotations

import numpy as np
import torch


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
        from diffusion.inference_timesteps import build_inference_timesteps
        from diffusion.schedules import get_beta_schedule

        beta = get_beta_schedule("linear", 1000)
        ac = np.cumprod(1.0 - beta)
        for name in ("ays", "ays_dit"):
            idx = build_inference_timesteps(name, 1000, 28, ac)
            assert idx.shape[0] == 28
            assert np.all(np.diff(idx.astype(np.int64)) < 0), f"{name} must be strictly descending"
            assert int(idx[0]) > int(idx[-1])

    def test_ays_dit_registered(self):
        from diffusion.inference_timesteps import list_timestep_schedules

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
    def test_euler_then_ab2(self):
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
    def test_predict_then_correct(self):
        from diffusion.solvers import SolverState, vp_unipc_update

        B = 1
        x = torch.randn(B, 2, 4, 4)
        x0 = torch.randn_like(x)
        state = SolverState(max_order=3)
        ab_c = torch.tensor([0.25])
        ab_n = torch.tensor([0.55])
        x_pred, state, need = vp_unipc_update(
            x=x,
            x0_pred=x0,
            alpha_bar_cur=ab_c,
            alpha_bar_next=ab_n,
            state=state,
            corrector_x0=None,
        )
        assert need is True
        x_corr, state, need2 = vp_unipc_update(
            x=x,
            x0_pred=x0,
            alpha_bar_cur=ab_c,
            alpha_bar_next=ab_n,
            state=state,
            corrector_x0=x0,
        )
        assert need2 is False
        assert x_pred.shape == x_corr.shape
