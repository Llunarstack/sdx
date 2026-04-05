# Native Stack

SDX includes optional native components for speed and metrics fidelity.

## Where native code lives

- `native/c/`: C metrics primitives.
- `native/cpp/`: C++ metrics and CUDA kernels.
- `native/rust/`: fast CLI tools for metrics and validation.
- `native/python/sdx_native/`: Python ctypes wrappers and tool discovery.
- `utils/native/`: compatibility shims used by higher-level modules.

## What it accelerates

- image luma/exposure stats
- clipping ratio detection
- Laplacian sharpness metrics
- connected-component counting for simple object heuristics

These metrics are consumed by test-time pick and benchmark scoring paths.

## Runtime behavior

- Core SDX still works without native libs.
- If native libs are available, wrappers in `sdx_native` are used.
- Fallback Python paths are used when native binaries/libs are absent.

## Operational checks

- Use `startup_readiness` for native stack visibility.
- Use `pretrained_status` for model-side wiring checks.
- Keep native binaries versioned with code changes that depend on them.
