# `diffusion/` layout

| Path | Role |
|------|------|
| **`gaussian_diffusion.py`** | `GaussianDiffusion`, `create_diffusion`, training losses, sampling (`sample_loop`: timestep schedule + solver). |
| **`inference_timesteps.py`** | Pluggable **inference** timestep indices (`ddim`, `euler`, `karras_rho`, `snr_uniform`, `quad_cosine`). |
| **`spectral_sfp.py`** | Optional FFT-weighted training loss (prototype). |
| **`schedules.py`** | VP β schedules (`get_beta_schedule`). |
| **`losses/`** | Timestep loss weights — `loss_weighting.py`, `timestep_loss_weight.py`. |
| **`loss_weighting.py`** | Shim → `losses.loss_weighting` (stable import path). |
| **`timestep_loss_weight.py`** | Shim → `losses.timestep_loss_weight`. |
| **`timestep_sampling.py`** | Training-time `t` distributions. |
| **`snr_utils.py`** | NumPy SNR / ᾱ helpers for analysis. |
| **`respace.py`**, **`sampling_utils.py`** | Respacing, thresholding. |
| **`cascaded_multimodal_pipeline.py`** | Optional cascaded scaffold (not default `train.py`). |

Add new **noise schedules** in `schedules.py`. Add new **loss weighting modes** under `losses/`.
