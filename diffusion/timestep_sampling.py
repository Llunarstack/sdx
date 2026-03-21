"""
Training-time timestep *distributions* (recent diffusion / flow practice).

Stable Diffusion 3 and many rectified-flow pipelines sample a continuous normalized
time ``u ~ sigmoid(Normal(mu, sigma))`` so training spends more capacity on some
noise regimes than i.i.d. uniform discrete ``t``. This repo still trains **VP
DDPM** (``GaussianDiffusion``); we only change **which integer indices** ``t``
are drawn — no change to ``q_sample`` or the noise schedule itself.

References (for further reading; not all apply 1:1 to VP-DDPM):
  - SD3 / diffusers: logit-normal timestep sampling and loss density (GitHub issues #9056, #8591).
  - Rectified flow & flow matching: continuous-time straight paths (training would need a different objective).
  - TPC, Rectified-CFG++, RF-Sampling: mostly inference or flow-specific — see ``docs/MODERN_DIFFUSION.md``.
"""

from __future__ import annotations

import torch


def sample_training_timesteps(
    batch_size: int,
    num_timesteps: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    mode: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """
    Sample a batch of diffusion step indices in ``[0, num_timesteps - 1]``.

    Args:
        batch_size: Number of samples.
        num_timesteps: ``diffusion.num_timesteps`` (e.g. 1000).
        device: Torch device for the output tensor.
        dtype: Integer dtype for indices (``torch.long`` for model ``t``).
        mode:
            - ``uniform``: same as ``torch.randint(0, T, (B,))`` (classic DDPM).
            - ``logit_normal``: sample ``u ~ sigmoid(N(logit_mean, logit_std))``, then
              ``t = round(u * (T-1))`` (SD3-style *discrete* analogue).
            - ``high_noise``: ``u ~ Beta(2, 1)`` on ``[0,1]`` then map to ``t`` — more
              weight on **high** noise (large ``t``), useful if you want extra
              steps learning heavily corrupted latents.
        logit_mean / logit_std: Gaussian parameters for ``logit_normal`` mode.
            Defaults ``0, 1`` match common SD3-style presets on the normalized axis.

    Returns:
        Tensor of shape ``(batch_size,)`` with integer timesteps.
    """
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    T = int(num_timesteps)
    B = int(batch_size)
    m = str(mode or "uniform").strip().lower()

    if m == "uniform":
        return torch.randint(0, T, (B,), device=device, dtype=dtype)

    if m in ("logit_normal", "logit-normal", "sd3"):
        z = torch.randn(B, device=device, dtype=torch.float32) * float(logit_std) + float(logit_mean)
        u = torch.sigmoid(z)
        t = (u * float(T - 1)).round().to(dtype=torch.long)
        return t.to(dtype=dtype)

    if m in ("high_noise", "high-noise", "beta_high"):
        dist = torch.distributions.Beta(
            torch.tensor(2.0, device=device),
            torch.tensor(1.0, device=device),
        )
        u = dist.sample((B,))
        t = (u * float(T - 1)).round().to(dtype=torch.long)
        return t.to(dtype=dtype)

    raise ValueError(f"Unknown timestep_sample_mode {mode!r}; use uniform | logit_normal | high_noise")
