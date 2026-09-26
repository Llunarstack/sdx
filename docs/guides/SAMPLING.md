# Sampling guide (solvers & schedules)

SDX separates **when** to evaluate the model (schedules) from **how** to update
latents (solvers). Guidance extras (CFG++, APG, dynamic threshold) compose with
any of the combinations below.

## Recommended combos

| Goal | VP (classic DiT) | Flow-matching ckpt |
|------|------------------|--------------------|
| **Best quality** | `--scheduler ays_dit --solver dpmpp_2m --steps 28` | `--flow-matching-sample --flow-solver dpmpp_2m --flow-schedule ays --steps 20` |
| **Fast** | `--scheduler ays_dit --solver unipc --steps 15` | `--flow-solver heun --flow-schedule karras --steps 12` |
| **Legacy** | `--scheduler ddim --solver ddim` | `--flow-solver euler --flow-schedule linear` |

Presets (`--preset sdxl|flux|anime|superior|fast`) soft-apply these defaults.
OP modes: `--op-mode quality` / `--op-mode fast`.

## VP solvers (`--solver`)

| Name | Notes |
|------|--------|
| `dpmpp_2m` | **Default.** Real DPM-Solver++ 2M (x0 / data-prediction form). |
| `dpmpp_3m` | Extra history term; slightly better at high NFE. |
| `unipc` | Predictor–corrector; strong at 12–20 steps (2 evals/step). |
| `heun` | Heun-on-DDIM (2 evals/step). |
| `ddim` | Classic deterministic DDIM. |

Aliases such as `dpm++_2m`, `dpmsolver_pp_1`, `uni_pc` resolve to the real backends
(they no longer silently map to DDIM).

## VP schedules (`--scheduler`)

| Name | Notes |
|------|--------|
| `ays_dit` | **Default.** Align-Your-Steps style with denser mid-SNR for DiT/T5. |
| `ays` | Classic AYS-inspired log-SNR knots. |
| `karras_rho` | σ-space ρ spacing (`--karras-rho`). |
| `snr_uniform` / `quad_cosine` / `ddim` / `euler` | Older grids. |

## Flow solvers (`--flow-solver`)

| Name | Notes |
|------|--------|
| `dpmpp_2m` | **Default for flow.** Adams–Bashforth 2 on `dx/ds = v`. |
| `heun` / `midpoint` | 2nd-order; Heun costs 2 evals/step. |
| `euler` | 1st-order baseline. |

## Flow schedules (`--flow-schedule`)

| Name | Notes |
|------|--------|
| `ays` | **Default.** AYS knots mapped onto continuous `s ∈ [1,0]`. |
| `ays_dit` | Extra mid-band samples. |
| `karras` | Densifies near clean (`s→0`). |
| `linear` | Uniform `linspace(1→0)`. |

## Implementation

- Solvers: [`diffusion/solvers/`](../../diffusion/solvers/)
- Schedules: [`diffusion/inference_timesteps.py`](../../diffusion/inference_timesteps.py)
- Loop wiring: [`diffusion/gaussian_diffusion.py`](../../diffusion/gaussian_diffusion.py)

```bash
python sample.py --ckpt ... --prompt "..." --scheduler ays_dit --solver dpmpp_2m --steps 28
python sample.py --ckpt ... --prompt "..." --flow-matching-sample --flow-solver dpmpp_2m --flow-schedule ays
```
