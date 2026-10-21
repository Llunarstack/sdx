"""Continuous s-grids for rectified-flow sampling."""

from __future__ import annotations

import numpy as np

from diffusion.inference_timesteps import EDM_SIGMA_MAX, EDM_SIGMA_MIN, ays_progress_knots, karras_sigmas
from diffusion.schedules import inverse_time_shift, time_shift
from diffusion.solvers.apex import build_apex_flow_s_grid


def build_flow_s_grid(
    name: str,
    num_steps: int,
    *,
    s_start: float = 1.0,
    s_end: float = 0.0,
    karras_rho: float = 7.0,
    shift: float = 1.0,
) -> np.ndarray:
    """``num_steps + 1`` continuous flow times from ``s_start`` to ``s_end``.

    ``linear`` is uniform in s; ``karras`` is EDM rho-spacing in sigma = s/(1-s); ``ays`` / ``ays_dit`` place
    Align-Your-Steps knots in s; ``apex`` is the APEX three-phase density (``diffusion.solvers.apex``).

    ``shift`` != 1 applies the SD3 resolution shift (``diffusion.schedules.time_shift``): the schedule runs
    between the unshifted endpoints and is then shifted, so the endpoints stay ``s_start`` / ``s_end``.
    """
    if float(shift) != 1.0:
        a = float(shift)
        base = build_flow_s_grid(
            name,
            num_steps,
            s_start=inverse_time_shift(float(s_start), a),
            s_end=inverse_time_shift(float(s_end), a),
            karras_rho=karras_rho,
        )
        out = time_shift(base, a)
        out[0], out[-1] = float(s_start), float(s_end)
        return out
    n = max(1, int(num_steps))
    key = str(name).lower().strip()
    s0, s1 = float(s_start), float(s_end)
    if key in ("", "linear", "uniform"):
        return np.linspace(s0, s1, n + 1, dtype=np.float64)

    if key in ("karras", "karras_rho"):
        # EDM rho-ramp on the flow's EDM noise level sigma = s/(1-s); s=1 (sigma=inf) is capped at EDM sigma_max.
        hi = min(s0 / max(1.0 - s0, 1e-12), EDM_SIGMA_MAX)
        lo = s1 / max(1.0 - s1, 1e-12) if s1 > 0.0 else EDM_SIGMA_MIN
        if hi <= lo:
            return np.linspace(s0, s1, n + 1, dtype=np.float64)
        to_zero = s1 <= 0.0
        sig = karras_sigmas(hi, lo, n if to_zero else n + 1, rho=float(karras_rho))
        s = np.append(sig / (1.0 + sig), [0.0] if to_zero else [])
        s[0], s[-1] = s0, s1
        return s

    if key in ("ays", "ays_dit"):
        knots = ays_progress_knots(key)
        prog = np.interp(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, len(knots)), knots)
        return (s0 + (s1 - s0) * prog).astype(np.float64)

    if key == "apex":
        return build_apex_flow_s_grid(n, s_start=s0, s_end=s1, rho=float(karras_rho))

    raise ValueError(f"Unknown flow schedule {name!r}; use {', '.join(list_flow_schedules())}")


def list_flow_schedules() -> tuple[str, ...]:
    return ("linear", "karras", "ays", "ays_dit", "apex")
