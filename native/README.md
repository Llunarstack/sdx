# SDX native helpers (Rust, Zig, C++, Go)

Small **high-throughput utilities** around the Python training stack. They are **optional**: `train.py` / `sample.py` do not depend on them, but they speed up dataset QA and give reusable math/code for extensions (e.g. custom preprocessors, bindings).

| Component | Role |
|-----------|------|
| **Rust** `rust/sdx-jsonl-tools` | Stream a manifest JSONL: validate JSON, require `image_path`/`caption` (or aliases), stats, prompt lint. |
| **Zig** `zig/sdx-linecrc` | Streaming **FNV-1a 64-bit** fingerprint over lines (detect manifest changes without loading Python). |
| **C++** `cpp/` | `libsdx_latent` — DiT/VAE **latent grid** helpers (`image_size`, `vae_scale`, `patch_size`) with **C ABI** for ctypes / other FFI. |
| **Go** `go/sdx-manifest` | Merge multiple JSONL files; optional dedupe by image path (first wins). |
| **Node** `js/sdx-jsonl-stat.mjs` | Zero-build manifest stats if you already have **Node 18+** (no Rust install). |
| **Node** `js/sdx-promptlint.mjs` | Zero-build prompt adherence lint for JSONL (pos/neg overlap, empty captions, token heuristics). |

## Build (quick)

### Rust
```bash
cd native/rust/sdx-jsonl-tools
cargo build --release
# Windows: target/release/sdx-jsonl-tools.exe
```

### Zig (0.13+)
```bash
cd native/zig/sdx-linecrc
zig build -Doptimize=ReleaseFast
# zig-out/bin/sdx-linecrc (or zig-out/bin/sdx-linecrc.exe)
```

### C++
```bash
cd native/cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# shared library: build/libsdx_latent.so or sdx_latent.dll + import lib on MSVC
```

### Go
```bash
cd native/go/sdx-manifest
go build -o sdx-manifest .
```

### Node (optional)
```bash
node native/js/sdx-jsonl-stat.mjs data/manifest.jsonl
node native/js/sdx-promptlint.mjs data/manifest.jsonl
```

## Use with SDX manifests

Manifest lines should be JSON objects with at least:
- `image_path` (or `path` / `image`)
- `caption` (or `text`)

Same conventions as `data/t2i_dataset.py` and `scripts/tools/data_quality.py`.

### Example: Rust validate + stats
```bash
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools stats data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools validate --min-caption-len 5 data/manifest.jsonl
native/rust/sdx-jsonl-tools/target/release/sdx-jsonl-tools prompt-lint --max-caption-tokens 250 data/manifest.jsonl
```

### Example: Zig fingerprint (pipe or file)
```bash
native/zig/sdx-linecrc/zig-out/bin/sdx-linecrc --file data/manifest.jsonl
type data\manifest.jsonl | native\zig\sdx-linecrc\zig-out\bin\sdx-linecrc.exe
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
```

### Example: merge JSONL (Go)
```bash
./sdx-manifest merge a.jsonl b.jsonl -o merged.jsonl --dedupe-key image_path
```

## Python integration (repo root on `PYTHONPATH`)

| Location | Role |
|----------|------|
| **`native/python/sdx_native/`** | **Source of truth:** `latent_geometry.py`, `native_tools.py` (ctypes, CLI discovery, FNV, merge). |
| **`utils/latent_geometry.py`** · **`utils/native_tools.py`** | Thin shims (add `native/python` to `sys.path`) so existing `from utils.native_tools import …` keeps working. |
| **`pyproject.toml`** | Pytest **`pythonpath`** includes `native/python` so `import sdx_native` works in tests. |

**Wired scripts**

- **`scripts/tools/data_quality.py`** — `--native-preflight` (Rust `stats` before filter), `--native-stats` (stats only), `--native-validate` (strict Rust validate).
- **`scripts/tools/op_preflight.py`** — `--native-manifest-check` (Rust `stats` to stderr before coverage scan).
- **`scripts/tools/dit_variant_compare.py`** — prints **patch token count** via `libsdx_latent` or Python math (`--vae-scale`).
- **`scripts/tools/jsonl_merge.py`** — merge manifests; prefers Go **`sdx-manifest`** if built.
- **`scripts/tools/quick_test.py`** — `--show-native` prints discovery JSON (paths empty until you build tools).

## Why these languages?

- **Rust**: safe parallel I/O–friendly CLI tooling, easy distribution as one binary.
- **Zig**: tiny, fast, predictable builds; good for checksum/fingerprint pipes.
- **C++**: stable **C ABI** shared library for grid math or future SIMD without pulling the full PyTorch stack.
- **Go**: simple concurrent merges and ops on huge text files with minimal dependencies.
