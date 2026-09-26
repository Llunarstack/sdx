"""Flow ODE helpers: midpoint solver + continuous s-grids for RF sampling."""

from __future__ import annotations

import numpy as np
import torch


def flow_midpoint_update(
    *,
    x: torch.Tensor,
    velocity: torch.Tensor,
    s_cur: float,
    s_next: float,
    model_fn,
    t_next_batch: torch.Tensor,
    step_idx: int,
) -> torch.Tensor:
    """
    Explicit midpoint (RK2): evaluate v at midpoint in s, one extra model call.

    ``model_fn(x, t_batch, step_idx) -> velocity``.
    """
    ds = float(s_next) - float(s_cur)
    if abs(ds) < 1e-12:
        return x
    s_mid = 0.5 * (float(s_cur) + float(s_next))
    x_mid = x + velocity * (s_mid - float(s_cur))
    v_mid = model_fn(x_mid, t_next_batch, step_idx)
    return x + v_mid * ds


def build_flow_s_grid(
    name: str,
    num_steps: int,
    *,
    s_start: float = 1.0,
    s_end: float = 0.0,
    karras_rho: float = 7.0,
) -> np.ndarray:
    """
    Build ``num_steps + 1`` values of continuous flow time s from ``s_start`` → ``s_end``.

    Schedules:
      - linear: uniform in s
      - karras: denser near clean (s→0), ρ-power in a sigma-like remap
      - ays / ays_dit: Align-Your-Steps style knots in s
    """
    n = max(1, int(num_steps))
    key = str(name).lower().strip()
    s0 = float(s_start)
    s1 = float(s_end)
    if key in ("", "linear", "uniform"):
        return np.linspace(s0, s1, n + 1, dtype=np.float64)

    if key in ("karras", "karras_rho"):
        # Map s through a Karras-like ramp so early (noisy) steps are larger.
        rho = float(karras_rho)
        u = np.linspace(0.0, 1.0, n + 1)
        # Invert typical Karras: more resolution near s=0 (clean).
        w = u**rho
        return (s0 + (s1 - s0) * w).astype(np.float64)

    if key in ("ays", "ays_dit"):
        # Knots in normalized progress (0=noise, 1=clean), then map to s.
        # AYS SDXL-inspired progressive densities; ays_dit densifies mid-band.
        if key == "ays":
            knots = np.array(
                [0.0, 0.05, 0.12, 0.22, 0.35, 0.50, 0.65, 0.78, 0.88, 0.94, 0.98, 1.0],
                dtype=np.float64,
            )
        else:
            # DiT/T5: extra mid-SNR samples (progress 0.3–0.7).
            knots = np.array(
                [
                    0.0,
                    0.04,
                    0.10,
                    0.18,
                    0.28,
                    0.38,
                    0.48,
                    0.58,
                    0.68,
                    0.78,
                    0.88,
                    0.94,
                    0.98,
                    1.0,
                ],
                dtype=np.float64,
            )
        # Progress 0 → s_start (noise), progress 1 → s_end (clean).
        prog = np.interp(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, len(knots)), knots)
        return (s0 + (s1 - s0) * prog).astype(np.float64)

    raise ValueError(f"Unknown flow schedule {name!r}; use linear, karras, ays, ays_dit")


def list_flow_schedules() -> tuple[str, ...]:
    return ("linear", "karras", "ays", "ays_dit")
