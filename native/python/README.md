# Python bridge (`sdx_native`)

Package **`sdx_native`** lives under **`native/python/sdx_native/`** so native helpers stay next to Rust/Zig/Go/C++ sources.

| Module | Role |
|--------|------|
| **`latent_geometry.py`** | Latent / DiT patch token math (matches `native/cpp` C ABI). |
| **`native_tools.py`** | Discover built CLIs, run JSONL tools, FNV fingerprints, ctypes `libsdx_latent`, JSONL merge. |

**Imports**

- Preferred (with repo root on `PYTHONPATH` and `native/python` discoverable): `from sdx_native.native_tools import native_stack_status`
- Stable alias: `from utils.native_tools import native_stack_status` (thin shim adds `native/python` to `sys.path`)

Pytest adds `native/python` via **`pyproject.toml`** (`pythonpath`).
