# Sampling guide (solvers & schedules)

SDX separates **when** to evaluate the model (schedules) from **how** to update
latents (solvers). The guidance stack (APG, FDG, CFG rescale, PAG, ...) and dynamic
thresholding compose with any of the combinations below.

## Recommended combos

| Goal | VP (classic DiT) | Flow-matching ckpt |
|------|------------------|--------------------|
| **Best quality** | `--scheduler ays_dit --solver dpmpp_2m --steps 28` | `--flow-matching-sample --flow-solver dpmpp_2m --flow-schedule ays --steps 20` |
| **More texture** | `--solver dpmpp_2m_sde --steps 30` (add `--refine restart`) | `--flow-solver dpmpp_2m_sde --sde-band 0.11,2.3` |
| **Fast** | `--scheduler ays_dit --solver unipc --steps 15` | `--flow-solver heun --flow-schedule karras --steps 12` |
| **Legacy** | `--scheduler ddim --solver ddim` | `--flow-solver euler --flow-schedule linear` |

Presets (`--preset sdxl|flux|anime|superior|fast`) soft-apply these defaults.
OP modes: `--op-mode quality` / `--op-mode fast`.

## VP solvers (`--solver`)

| Name | Notes |
|------|--------|
| `dpmpp_2m` | **Default.** DPM-Solver++(2M), data-prediction exponential integrator (Lu et al. 2022). |
| `dpmpp_2m_sde` | DPM-Solver++(2M) SDE: re-injects noise each step (`--eta`, default 1), which corrects accumulated ODE error and keeps fine texture. |
| `dpmpp_3m` | Third-order multistep; better at high NFE, can overshoot below ~12 steps. |
| `unipc` | UniPC-2 (Zhao et al. 2023): DPM++ predictor + UniC corrector, one forward per step; strong at 10–20 steps. |
| `heun` | EDM Heun on the noise-prediction ODE (2 forwards/step); needs a dense schedule (e.g. `karras_rho`). |
| `ddim` | DDIM; `--eta` > 0 gives DDPM-like ancestral noise. |
| `apex` / `apex_adaptive` | Embedded-error multistep (see `docs/research/APEX_SAMPLER.md`). |

Aliases such as `dpm++_2m`, `dpmsolver_pp_1`, `uni_pc` resolve to the real backends.

## Noise re-injection

- `--eta` applies to `ddim`, `heun` and `dpmpp_2m_sde` (VP) and `dpmpp_2m_sde` (flow).
  ODE solvers reject `--eta > 0` instead of ignoring it.
- `--sde-band LO,HI` injects noise only where the EDM noise level σ lies in `[LO, HI]`
  (VP σ = sqrt((1-ᾱ)/ᾱ), flow σ = s/(1-s)); `0.11,2.3` is flow s ∈ [0.1, 0.7], where texture forms.
- Frontier serendipity curves (`step_noise_scales`) multiply η per step and need a stochastic solver.

## Exposure-bias correction

`--eps-scale 1.003` (optionally with `--eps-scale-k`) divides the noise prediction by
λ_t = eps_scale + eps_scale_k·t at every evaluation (epsilon scaling, Ning et al., ICLR 2024).
Off by default; it costs no extra forwards.

## Refinement (`--refine`)

Off by default. `--refine restart` runs Restart sampling (Xu et al., NeurIPS 2023) on the
result: renoise to `--refine-t` (default 200), re-solve with the same sampler in
`--restart-steps` steps, `--restart-cycles` times (default 2). It works for VP and flow
checkpoints. `--refine-gate auto` skips it when a quick quality score is already high.

## VP schedules (`--scheduler`)

| Name | Notes |
|------|--------|
| `ays_dit` | **Default.** Align-Your-Steps style with denser mid-SNR for DiT/T5. |
| `ays` | Classic AYS-inspired log-SNR knots. |
| `karras_rho` | EDM (Karras et al. 2022) ρ-spacing in σ (`--karras-rho`, default 7). |
| `snr_uniform` / `quad_cosine` / `ddim` / `euler` | Older grids. |

## Flow solvers (`--flow-solver`)

| Name | Notes |
|------|--------|
| `dpmpp_2m` | **Default for flow.** DPM-Solver++(2M) with α = 1−s, σ = s; the first step is Euler. |
| `dpmpp_2m_sde` | Its SDE variant (`--eta`, `--sde-band`). |
| `heun` / `midpoint` | 2nd-order, 2 forwards/step; Heun takes an Euler last step to s = 0 (EDM). |
| `euler` | 1st-order baseline. |

## Flow schedules (`--flow-schedule`)

| Name | Notes |
|------|--------|
| `ays` | **Default.** AYS knots mapped onto continuous `s ∈ [1,0]`. |
| `ays_dit` | Extra mid-band samples. |
| `karras` | EDM ρ-spacing in σ = s/(1-s), σ_max capped at 80. |
| `linear` | Uniform `linspace(1→0)`. |

## Resolution, aspect ratio and the timestep shift

- **Canvas.** `--width/--height` set the latent grid, not a post-decode stretch. The requested size is
  snapped to whole DiT patches (16 px for patch 2) when its area is within 2× of the native square;
  otherwise sampling runs at the native pixel area with the requested aspect ratio and the last step
  (`--hires-fix`, else a resize) reaches the size. Without a size, img2img follows the init image's
  aspect ratio. The DiTs take any patch grid: positions use PixArt-α interpolated sin/cos (the stored
  table at the native grid, so old checkpoints are unchanged), `use_rope` is axial 2D RoPE, and
  block-AR masks are built per grid.
- **Shift.** Flow grids get the SD3 shift `s' = a·s / (1 + (a−1)·s)` (Esser et al. 2024, Eq. 23) with
  `a = --flow-shift · sqrt(tokens / native tokens)`; `--flow-shift` defaults to the checkpoint's
  `flow_shift`, else 1, so the native grid is unchanged. `diffusion.schedules` holds the helpers
  (`time_shift`, `sd3_resolution_shift`, `flux_resolution_shift`, `vp_shifted_timestep`) for training
  to reuse.
- **Hires-fix / dual-stage.** Both work for VP and flow. The upscaled latent is re-noised to
  `--hires-strength` (or `--dual-detail-strength`) shifted for its grid (flow `time_shift`, VP: the
  timestep whose log-SNR is lower by `2·log a`), then denoised with the run's solver settings.

## Dynamic thresholding

`--dynamic-threshold-percentile 99.5` is Imagen's dynamic thresholding of the x0 prediction
at every step: clip to s = max(percentile(|x0|), r) and rescale onto [−r, r], where r is
`--dynamic-threshold-value` (default 1, the pixel range). For VAE latents set r to the
latent range (e.g. 4); with r = 1 the latent is shrunk toward grey.

## Implementation

- Solvers: [`diffusion/solvers/`](../../diffusion/solvers/)
- Schedules: [`diffusion/inference_timesteps.py`](../../diffusion/inference_timesteps.py)
- Loop wiring: [`diffusion/gaussian_diffusion.py`](../../diffusion/gaussian_diffusion.py)
- Resolution shift: [`diffusion/schedules.py`](../../diffusion/schedules.py); canvas: [`diffusion/pixel_perfect.py`](../../diffusion/pixel_perfect.py)

```bash
python sample.py --ckpt ... --prompt "..." --scheduler ays_dit --solver dpmpp_2m --steps 28
python sample.py --ckpt ... --prompt "..." --solver dpmpp_2m_sde --refine restart
python sample.py --ckpt ... --prompt "..." --flow-matching-sample --flow-solver dpmpp_2m --flow-schedule ays
```
