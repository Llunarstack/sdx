# C++ — `libsdx_latent`

Shared library with a **C ABI** for latent grid / DiT patch token math.

- **Headers:** `include/sdx/latent.h`
- **Sources:** `src/sdx_latent.cpp`
- **Build:** `cmake -S . -B build && cmake --build build`

Python loads the built DLL/.so via `sdx_native.native_tools` (ctypes).
