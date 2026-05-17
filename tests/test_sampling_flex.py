"""Sampler / timestep / CFG schedule composition (torch-free where possible)."""

from __future__ import annotations

import numpy as np


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
    from diffusion.inference_timesteps import build_inference_timesteps

    ac = np.clip(np.linspace(0.999, 0.01, 200), 1e-4, 1.0 - 1e-4)
    out = build_inference_timesteps("indices:199,160,120,80,40,10,0", 200, 7, ac)
    assert out.shape == (7,)
    assert out[0] >= out[-1]


def test_snr_cfg_multiplier_bounded() -> None:
    from diffusion.cfg_schedulers import cfg_scale_snr_aware_multiplier

    a_hi = cfg_scale_snr_aware_multiplier(0.995)
    a_lo = cfg_scale_snr_aware_multiplier(0.02)
    assert 0 < a_hi < 10 and 0 < a_lo < 10
