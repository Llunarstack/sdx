# SDX native helpers (Rust, Zig, C++, CUDA, Go, Mojo)

**Broader map (ecosystem libs + how they help quality / training / adherence):** [docs/NATIVE_AND_SYSTEM_LIBS.md](../docs/NATIVE_AND_SYSTEM_LIBS.md).

Small **high-throughput utilities** around the Python training stack. They are **optional**: `train.py` / `sample.py` do not depend on them, but they speed up dataset QA and give reusable math/code for extensions (e.g. custom preprocessors, bindings).

| Component | Role |
|-----------|------|
| **Rust** `rust/sdx-jsonl-tools` | JSONL: **stats**, **validate**, **prompt-lint**, **`image-paths`**, **`dup-image-paths`**, **`file-fnv`**, **`caption-len-buckets`**. |
| **Rust** `rust/sdx-noise-schedule` | CLI: **linear** / **cosine** VP-DDPM schedule tables (CSV) for analysis vs `diffusion/`. |
| **Rust** `rust/sdx-image-metrics` | CLI: **stats** (mean luma, clip ratio, Laplacian variance) and **count-blobs** connected components for quick count heuristics. |
| **Zig** `zig/sdx-linecrc` | Streaming **FNV-1a 64-bit** fingerprint over file bytes (manifest change detection). |
| **Zig** `zig/sdx-pathstat` | Given a **newline-separated path list**, print `path<TAB>size_bytes<TAB>ok|missing` (fast file-exists + size audit; pair with Rust `image-paths`). |
| **C++** `cpp/` | `libsdx_latent`, **`sdx_line_stats`**, **`sdx_fnv64_file`**, **`sdx_image_metrics`**, inference/beta helpers — **C ABI** for ctypes. **`include/sdx/experimental/`**: **tensor_lite**, **vram_pool_stub**, **augmentor_plugin** (header stubs). |
| **CUDA** `cpp/cuda/` + [cuda/README.md](cuda/README.md) | Optional **`sdx_cuda_hwc_to_chw`**, **`sdx_cuda_ml`**, **`sdx_cuda_flow_matching`**, **`sdx_cuda_nf4`**, **`sdx_cuda_sdpa_online`**, **`sdx_cuda_image_metrics`**; `-DSDX_BUILD_CUDA=ON`. |
| **Mojo** `mojo/` + [mojo/README.md](mojo/README.md) | Optional **Modular Mojo** stubs + **Python** `mojopy` launcher for CPU/SIMD experiments. |
| **Go** `go/sdx-manifest` | Merge multiple JSONL files; optional dedupe by image path (first wins). |
| **Python** `sdx_native.jsonl_manifest_pure` | Zero-build manifest **stats** + **prompt-lint** (same role as the old `js/*.mjs`; no Node). |

## Build (quick)

### Rust
```bash
cd native/rust/sdx-jsonl-tools
cargo build --release
# Windows: target/release/sdx-jsonl-tools.exe

cd native/rust/sdx-noise-schedule
cargo build --release
# target/release/sdx-noise-schedule linear --steps 1000

cd native/rust/sdx-image-metrics
cargo build --release
# target/release/sdx-image-metrics stats --image sample.png
```

### Zig (0.13+)
```bash
cd native/zig/sdx-linecrc
zig build -Doptimize=ReleaseFast
# zig-out/bin/sdx-linecrc (or .exe)

cd native/zig/sdx-pathstat
zig build -Doptimize=ReleaseFast
# zig-out/bin/sdx-pathstat — stat paths from a file list
```

### C++
```bash
cd native/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# shared library: build/libsdx_latent.so or sdx_latent.dll + import lib on MSVC
# optional CUDA kernel (requires nvcc):
# cmake -S . -B build -DSDX_BUILD_CUDA=ON
# cmake --build build --config Release
```

### Go
```bash
cd native/go/sdx-manifest
go build -o sdx-manifest .
```

### Python JSONL stat / prompt-lint (no install)
```bash
python -m sdx_native.jsonl_manifest_pure stat data/manifest.jsonl
python -m sdx_native.jsonl_manifest_pure promptlint data/manifest.jsonl
```

### One-shot native build (C++ + optional CUDA + Rust)
```powershell
.\scripts\tools\native\build_native.ps1
# CPU-only C++:
#   $env:SDX_BUILD_CUDA='0'; .\scripts\tools\native\build_native.ps1
```
The script passes ``-DSDX_BUILD_CUDA=ON|OFF`` to CMake with **quoted** arguments so the cache never stores a bogus literal. Invalid or mismatched ``SDX_BUILD_CUDA`` lines in ``CMakeCache.txt`` trigger an automatic ``build/`` delete.
```bash
bash scripts/tools/native/build_native.sh
```

### Mojo (Pixi, linux-64 / WSL2)
See [mojo/README.md](mojo/README.md) and `native/mojo/pixi.toml`. On Windows use `install_mojo_wsl.ps1` or WSL `pixi install` under `native/mojo`.

## Use with SDX manifests

Manifest lines should be JSON objects with at least:
- `image_path` (or `path` / `image`)
- `caption` (or `text`)

Same conventions as `data/t2i_dataset.py` and `scripts/tools/data/data_quality.py`.

### Example: Rust validate + stats
```bash
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools stats data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools validate --min-caption-len 5 data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools prompt-lint --max-caption-tokens 250 data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools image-paths --sort data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools dup-image-paths data/manifest.jsonl
```

### Example: Zig fingerprint (pipe or file)
```bash
native/zig/sdx-linecrc/zig-out/bin/sdx-linecrc --file data/manifest.jsonl
type data\manifest.jsonl | native\zig\sdx-linecrc\zig-out\bin\sdx-linecrc.exe
```

### Example: path list → sizes (Zig `sdx-pathstat`)
```bash
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools image-paths data/manifest.jsonl > paths.txt
native/zig/sdx-pathstat/zig-out/bin/sdx-pathstat --file paths.txt
```

### Example: C ABI from Python (`ctypes`)
```python
import ctypes
from pathlib import Path

dll = ctypes.CDLL(str(Path("native/cpp/build/Release/sdx_latent.dll")))  # adjust for your platform
dll.sdx_num_patch_tokens.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
dll.sdx_num_patch_tokens.restype = ctypes.c_int
n = dll.sdx_num_patch_tokens(256, 8, 2)  # 32x32 latent -> 256 patch tokens for patch 2
print(n)
dll.sdx_latent_numel.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
dll.sdx_latent_numel.restype = ctypes.c_int
print(dll.sdx_latent_numel(4, 32, 32))  # 4 * 32 * 32 latent elements
```

### Example: merge JSONL (Go)
```bash
./sdx-manifest merge a.jsonl b.jsonl -o merged.jsonl --dedupe-key image_path
```

## Python integration (repo root on `PYTHONPATH`)

| Location | Role |
|----------|------|
| **`native/python/sdx_native/`** | **Source of truth:** `latent_geometry.py`, `text_hygiene.py`, `native_tools.py` (ctypes, CLI discovery, FNV, merge). |
| **`utils/native/latent_geometry.py`** · **`utils/native/text_hygiene.py`** · **`utils/native/native_tools.py`** | Thin shims (add `native/python` to `sys.path`) so existing `from utils.native.native_tools import …` keeps working. |
| **`pyproject.toml`** | Pytest **`pythonpath`** includes `native/python` so `import sdx_native` works in tests. |

**Wired scripts**

- **`scripts/tools/data/data_quality.py`** — `--native-preflight` (Rust `stats` before filter), `--native-stats` (stats only), `--native-validate` (strict Rust validate).
- **`scripts/tools/data/caption_hygiene.py`** — JSONL **NFKC** samples, **caption fingerprint** dup report, pos/neg overlap (`sdx_native.text_hygiene`); pairs with `train.py --caption-unicode-normalize`.
- **`scripts/tools/ops/op_preflight.py`** — `--native-manifest-check` (Rust `stats` to stderr before coverage scan).
- **`scripts/tools/dit_variant_compare.py`** — prints **patch token count** via `libsdx_latent` or Python math (`--vae-scale`).
- **`scripts/tools/data/jsonl_merge.py`** — merge manifests; prefers Go **`sdx-manifest`** if built.
- **`scripts/tools/data/manifest_paths.py`** — **`image-paths`** / **`dup-image-paths`** (Rust); pipe to Zig **`sdx-pathstat`** for byte sizes.
- **`scripts/tools/dev/quick_test.py`** — `--show-native` prints discovery JSON (paths empty until you build tools).

## Why these languages?

- **Rust**: safe parallel I/O–friendly CLI tooling, easy distribution as one binary.
- **Zig**: tiny, fast, predictable builds; good for checksum/fingerprint pipes.
- **C++**: stable **C ABI** shared library for grid math or future SIMD without pulling the full PyTorch stack.
- **Go**: simple concurrent merges and ops on huge text files with minimal dependencies.
