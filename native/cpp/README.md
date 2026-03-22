# C++ — `libsdx_latent`

Shared library with a **C ABI** for latent grid / DiT patch token math.

- **Headers:** `include/sdx/latent.h`
- **Sources:** `src/sdx_latent.cpp`
- **Exports:** `sdx_latent_spatial_size`, `sdx_patch_grid_dim`, `sdx_num_patch_tokens`, `sdx_latent_hw`, **`sdx_latent_numel`** (channels×H×W for latent tensors)
- **Build:** `cmake -S . -B build && cmake --build build`

Python loads the built DLL/.so via `sdx_native.native_tools` (ctypes). Use `LatentLib.latent_numel(c,h,w)` for parity.
