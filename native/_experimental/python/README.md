# Python bridge (`sdx_native`)

Package **`sdx_native`** lives under **`native/_experimental/python/sdx_native/`** so native helpers stay next to Rust/Zig/Go/C++ sources. Every ctypes wrapper loads its library through `native_tools.load_shared_library(name)` and returns `None` (callers fall back to NumPy/PyTorch) when the library is not built.

| Module | Role |
|--------|------|
| **`native_tools.py`** | Discover built CLIs and shared libraries (`shared_library_path`, `load_shared_library`), run `sdx-jsonl-tools`, FNV fingerprints, ctypes `libsdx_latent`, JSONL merge, `native_stack_status()`. |
| **`latent_geometry.py`** | Latent / DiT patch token math (matches `native/cpp` C ABI). |
| **`text_hygiene.py`** | Caption **NFKC** + zero-width strip, fingerprint (SHA256 / optional **xxhash**), pos/neg overlap; `train.py --caption-unicode-normalize`. |
| **`jsonl_caption_hygiene.py`** | Per-row caption hygiene for `scripts.tools normalize_captions`. |
| **`beta_schedules_native.py`**, **`diffusion_math_native.py`**, **`inference_timesteps_native.py`** | Diffusion schedule math (C++ / Rust). |
| **`gaussian_blur_native.py`** | CUDA latent Gaussian blur (sampler). |
| **`mask_ops_native.py`** | C++ mask → patch weights (part-aware training). |
| **`image_metrics_native.py`** | C++ / CUDA luma, clipping, Laplacian variance, component count (test-time pick). |
| **`image_ops_native.py`** | Rust blob counting (count-adherence scorer). |
| **`quality_scorers_native.py`** | Rust highlight-clip fraction (`utils.ai_accel`). |
| **`style_ops_native.py`**, **`cuda_style_pick_native.py`** | Rust token Jaccard and CUDA cosine pick for `utils.prompt.style_native`. |

**Imports**

- With `native/_experimental/python` on `sys.path`: `from sdx_native.native_tools import native_stack_status`
- From repo code: `from utils.native import native_stack_status` (`utils/native` adds the path and resolves names lazily)

Pytest adds `native/_experimental/python` via **`pyproject.toml`** (`pythonpath`).
