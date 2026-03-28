/**
 * @file augmentor_plugin.hpp
 * C-style **plugin hook** for image augmentations (blur / crop / color jitter).
 *
 * Real pipelines often use **OpenCV**, **libvips**, or **torchvision** transforms in the dataloader.
 * A C ABI lets you hot-load `.dll` / `.so` augmentors from a thin Python `ctypes` wrapper.
 */
#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

/// RGB8 HWC, contiguous. `user` is opaque plugin state.
typedef int (*sdx_augment_rgb8_fn)(void *user, uint8_t *hwc, int height, int width, int channels);

struct SdxAugmentorPlugin {
    void *user;
    sdx_augment_rgb8_fn apply;
};

} // extern "C"
