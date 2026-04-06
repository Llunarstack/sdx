# Python bridge (`sdx_native`)

Package **`sdx_native`** lives under **`native/python/sdx_native/`** so native helpers stay next to Rust/Zig/Go/C++ sources.

| Module | Role |
|--------|------|
| **`latent_geometry.py`** | Latent / DiT patch token math (matches `native/cpp` C ABI). |
| **`text_hygiene.py`** | Caption **NFKC** + zero-width strip, fingerprint (SHA256 / optional **xxhash**), pos/neg overlap helper; `train.py --caption-unicode-normalize`. |
| **`line_stats_native.py`** | Optional **`sdx_line_stats`** DLL — fast manifest bytes + newline count. |
| **`fnv64_file_native.py`** | Optional **`sdx_fnv64_file`** DLL — streaming FNV-1a 64 (manifest fingerprint). |
| **`cuda_hwc_to_chw.py`** | Optional **`sdx_cuda_hwc_to_chw`** — uint8 HWC→float NCHW (see `native/cuda/README.md`). |
| **`cuda_l2_normalize.py`** | Optional **`sdx_cuda_ml`** — L2 row normalize for float32 matrices. |
| **`jsonl_manifest_pure.py`** | Pure-Python JSONL stat + prompt-lint (no native build). |
| **`native_tools.py`** | Discover built CLIs, run JSONL tools, FNV fingerprints, ctypes `libsdx_latent`, JSONL merge, stack status. |

**Imports**

- Preferred (with repo root on `PYTHONPATH` and `native/python` discoverable): `from sdx_native.native_tools import native_stack_status`
- Stable alias: `from utils.native import native_stack_status` (unified shim adds `native/python` to `sys.path`)

Pytest adds `native/python` via **`pyproject.toml`** (`pythonpath`).
