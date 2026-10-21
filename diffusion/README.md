# `diffusion/` layout

| Path | Role |
|------|------|
| **`gaussian_diffusion.py`** | `GaussianDiffusion`, `create_diffusion`, training losses, sampling (`sample_loop`, VP and flow loops). |
| **`solvers/`** | Inference ODE/SDE steppers (DPM++ 1/2/3M, UniPC, APEX, flow solvers). |
| **`inference_timesteps.py`** | Inference timestep grids (`ddim`, `euler`, `karras_rho`, `ays`, …). |
| **`cfg_schedulers.py`** | Per-step CFG scale schedules (used by the sampler). |
| **`sampling.py`** | Holy-grail recipe: per-step CFG/control/adapter plan, CADS, presets, knob guard. |
| **`sampling_utils.py`** | Thresholding, latent blur / unsharp. |
| **`schedules.py`** | VP β schedules (`get_beta_schedule`). |
| **`timestep_sampling.py`** | Training-time `t` distributions and per-timestep loss weights (min-SNR, EDM, v/eps). |
| **`flow_matching.py`** | Rectified-flow loss, flow-time sampling, velocity-consistency distillation loss. |
| **`snr_utils.py`** | SNR / ᾱ helpers and the SD3/Flux resolution time shift. |
| **`respace.py`** | DDIM-style timestep respacing. |
| **`pixel_perfect.py`** | Pixel/latent canvas snapping for a requested output size. |
| **`bridge_training.py`**, **`spectral_sfp.py`** | Optional training auxiliaries. |

Add new **noise schedules** in `schedules.py` and new **loss weighting modes** in `timestep_sampling.py`.
