/**
 * @file vram_pool_stub.hpp
 * **Stub** for a slab-style GPU allocator (pre-allocate arenas, suballocate blocks).
 *
 * Production training usually relies on **PyTorch caching allocator** + `empty_cache` tuning.
 * A custom pool can help **long runs** with fixed-size workspaces (e.g. fixed batch latents).
 *
 * This header documents the idea; integrate with **CUDA driver APIs** in a real `.cu` module.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace sdx::experimental {

class VramPoolStub {
public:
    /// Pretend to reserve `bytes` on device `ordinal` (no-op stub).
    bool reserve(std::size_t bytes, int device_ordinal = 0) {
        (void)bytes;
        (void)device_ordinal;
        reserved_ = true;
        return true;
    }

    bool reserved() const { return reserved_; }

private:
    bool reserved_{false};
};

} // namespace sdx::experimental
