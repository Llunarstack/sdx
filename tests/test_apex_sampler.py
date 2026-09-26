"""APEX sampler/scheduler unit tests."""

from __future__ import annotations

import numpy as np
import torch


class TestApexSchedule:
    def test_vp_indices_descending(self):
        from diffusion.inference_timesteps import build_inference_timesteps, list_timestep_schedules
        from diffusion.schedules import get_beta_schedule

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
        from diffusion.solvers.apex import redistribute_remaining_vp_indices

        beta = get_beta_schedule("linear", 1000)
        ac = np.cumprod(1.0 - beta)
        tail = redistribute_remaining_vp_indices(800, 10, 1000, ac, difficulty=2.0)
        assert len(tail) == 10
        assert int(tail[-1]) == 0
        assert np.all(np.diff(tail.astype(np.int64)) < 0)

    def test_phase_weight_peaks_mid(self):
        from diffusion.solvers.apex import phase_weight

        mid = phase_weight(0.42)
        early = phase_weight(0.05)
        late = phase_weight(0.95)
        assert mid > early
        assert mid > late
