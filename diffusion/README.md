# `diffusion/` layout

| Path | Role |
|------|------|
| **`gaussian_diffusion.py`** | `GaussianDiffusion`, `create_diffusion`, training losses, sampling (`sample_loop`, VP and flow loops), per-step CFG schedules, thresholding and latent blur / unsharp. |
| **`solvers.py`** | Inference ODE/SDE steppers (DPM++ 1/2/3M, UniPC, APEX), VP timestep grids (`ddim`, `euler`, `karras_rho`, `ays`, …) and flow s-grids. |
| **`schedules.py`** | VP β schedules (`get_beta_schedule`), SNR / ᾱ helpers, SD3/Flux resolution time shift, respacing, training-time `t` distributions and per-timestep loss weights (min-SNR, EDM, v/eps). |
| **`flow_matching.py`** | Rectified-flow loss, flow-time sampling, velocity-consistency distillation loss, Spectral Flow Prediction loss, latent-bridge auxiliary loss. |
| **`sampling.py`** | Holy-grail recipe (per-step CFG/control/adapter plan, CADS, presets, knob guard) and pixel/latent canvas snapping. |

Add new **noise schedules** and **loss weighting modes** in `schedules.py`, new **solvers** or **timestep grids** in `solvers.py`.
