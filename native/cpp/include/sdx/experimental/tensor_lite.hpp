/**
 * @file tensor_lite.hpp
 * Minimal **CPU** shape/strides view — sketch for a "tensor-lite" without LibTorch.
 * For real training, prefer **PyTorch** tensors; this is for learning / native glue code.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sdx::experimental {

struct TensorLiteF32 {
    std::vector<float> storage;
    std::vector<std::size_t> shape;

    std::size_t numel() const {
        std::size_t n = 1;
        for (std::size_t d : shape) {
            n *= d;
        }
        return n;
    }

    bool reshape(const std::vector<std::size_t> &new_shape) {
        std::size_t n = 1;
        for (std::size_t d : new_shape) {
            n *= d;
        }
        if (n != storage.size()) {
            return false;
        }
        shape = new_shape;
        return true;
    }
};

} // namespace sdx::experimental
