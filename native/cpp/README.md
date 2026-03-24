# C++ — native helpers (`libsdx_latent`, `sdx_inference_timesteps`, `sdx_beta_schedules`)

Optional shared libraries with a **C ABI** for Python (ctypes). Training and sampling still run on PyTorch; these cover small CPU-side math.

## `libsdx_latent`

Latent grid / DiT patch token math.

- **Headers:** `include/sdx/latent.h`
- **Sources:** `src/sdx_latent.cpp`
- **Exports:** `sdx_latent_spatial_size`, `sdx_patch_grid_dim`, `sdx_num_patch_tokens`, `sdx_latent_hw`, **`sdx_latent_numel`**

## `sdx_inference_timesteps`

Finalizes VP inference timestep index paths (clip, monotonic chain, resample / interpolate) to match `diffusion/inference_timesteps.py`. Loaded via `sdx_native.inference_timesteps_native` when present.

- **Header:** `include/sdx/inference_timesteps.h`
- **Source:** `src/sdx_inference_timesteps.cpp`
- **Export:** `sdx_it_finalize_path`

## `sdx_beta_schedules`

Squared-cosine v2 training betas (same as `diffusion.schedules.squared_cosine_beta_schedule_v2`). Loaded via `sdx_native.beta_schedules_native` when present.

- **Header:** `include/sdx/beta_schedules.h`
- **Source:** `src/sdx_beta_schedules.cpp`
- **Export:** `sdx_squared_cosine_betas_v2`

## Build

From `native/cpp`:

```bash
cmake -S . -B build && cmake --build build
```

This produces all targets. Python discovers DLLs under `build/Release` or `build/Debug` (Windows) or `.so` / `.dylib` in `build/` (see `sdx_native.native_tools`).
