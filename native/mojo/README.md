# Mojo (Modular) — optional high-performance kernels

[Mojo](https://docs.modular.com/mojo/) compiles to native code. This folder is **optional** for `train.py` / `sample.py`.

## Install (Pixi + Modular channel)

**Linux x86_64 / WSL2:** use the checked-in `pixi.toml` (lockfile: `pixi.lock`).

```bash
# Install Pixi: https://pixi.sh/latest/
cd native/mojo
pixi install
pixi run mojo-run   # prints stub line
```

**Windows (native):** there is **no** `mojo` package for **win-64** on `conda.modular.com`. Use **WSL2** or a Linux box.

```powershell
powershell -File native/mojo/install_mojo_wsl.ps1
```

**macOS ARM:** edit `pixi.toml` and add `"osx-arm64"` to `platforms`, then `pixi install`.

**Version note:** Conda publishes Mojo as **0.26.x**; the dependency must be `>=0.25.0` (PEP 440), not `>=25.0` (that would exclude 0.x releases).

## Layout

| Path | Role |
|------|------|
| `pixi.toml` / `pixi.lock` | Reproducible Mojo + compiler deps (linux-64 default). |
| `src/sdx_stub.mojo` | Minimal `mojo src/sdx_stub.mojo` smoke target. |
| `mojopy/launcher.py` | Python: run `mojo <file.mojo>` (falls back to `mojo run` if needed). |
| `install_mojo_wsl.ps1` | Windows helper → WSL `pixi install` + stub. |

## Python ↔ Mojo

- **Subprocess:** add `native/mojo` to `PYTHONPATH`, then `from mojopy.launcher import run_mojo_file` (still needs `mojo` on PATH, e.g. after `pixi shell`).
- **Embedded:** see [Modular Mojo install](https://docs.modular.com/mojo/manual/install) (Pixi / uv workflows).

## When to prefer other `native/` tools

| Need | Prefer |
|------|--------|
| JSONL stats / validate | Rust `sdx-jsonl-tools` or `sdx_native.jsonl_manifest_pure` |
| FNV / line CRC | Zig `sdx-linecrc` |
| Latent grid math | C++ `sdx_latent` |
| Image training on GPU | PyTorch + CUDA / `torch.compile` |

Mojo fits **new** CPU-bound kernels while keeping Python orchestration.
