# `diffusion/` layout

| Path | Role |
|------|------|
| **`gaussian_diffusion.py`** | `GaussianDiffusion`, `create_diffusion`, training losses, sampling (`sample_loop`). |
| **`inference_timesteps.py`** | Inference timestep grids (`ddim`, `euler`, `karras_rho`, …). |
| **`cfg_schedulers.py`** | Per-step CFG scale schedules (used by the sampler). |
| **`schedules.py`** | VP β schedules (`get_beta_schedule`). |
| **`losses/`** | Timestep loss weights — canonical import path. |
| **`sampling/`** | Holy Grail presets, runtime guards, advanced sampling helpers. |
| **`timestep_sampling.py`** | Training-time `t` distributions. |
| **`snr_utils.py`** | NumPy SNR / ᾱ helpers for analysis. |
| **`respace.py`**, **`sampling_utils.py`** | Respacing, thresholding. |
| **`flow_matching.py`**, **`bridge_training.py`**, **`spectral_sfp.py`** | Optional training auxiliaries. |
| **`cascaded_multimodal_pipeline.py`** | Optional cascaded scaffold (not default `train.py`). |

Add new **noise schedules** in `schedules.py`. Add new **loss weighting modes** under `losses/`. Prefer **`diffusion.sampling`** for Holy Grail / advanced sampling APIs.
