/**
 * CPU helpers for mask → patch-weight operations used in part-aware training.
 *
 * sdx_mask_to_patch_weights_f32:
 *   Downsample a (B, 1, H, W) float32 binary/soft mask to (B, ph*pw) by
 *   computing the mean of each (H/ph, W/pw) cell — equivalent to
 *   F.adaptive_avg_pool2d followed by flatten(1) in PyTorch, but without
 *   the Python/autograd overhead when called from the data-loading thread.
 *
 * C ABI so Python can call via ctypes without a build system dependency on PyTorch.
 */
#include "sdx/mask_ops.h"

#include <cstring>
#include <stdexcept>

extern "C" int sdx_mask_to_patch_weights_f32(
    const float *mask,   /* (B, 1, H, W) row-major */
    float       *out,    /* (B, ph * pw) row-major, caller-allocated */
    int B, int H, int W,
    int ph, int pw)
{
    if (!mask || !out || B <= 0 || H <= 0 || W <= 0 || ph <= 0 || pw <= 0)
        return -2;
    if (H % ph != 0 || W % pw != 0)
        return -3; /* dimensions must be divisible */

    const int cell_h = H / ph;
    const int cell_w = W / pw;
    const float inv_cell = 1.0f / (float)(cell_h * cell_w);

    for (int b = 0; b < B; ++b) {
        const float *plane = mask + (std::size_t)b * H * W;
        float       *row   = out  + (std::size_t)b * ph * pw;

        for (int py = 0; py < ph; ++py) {
            for (int px = 0; px < pw; ++px) {
                float sum = 0.0f;
                const int y0 = py * cell_h;
                const int x0 = px * cell_w;
                for (int dy = 0; dy < cell_h; ++dy) {
                    const float *src_row = plane + (y0 + dy) * W + x0;
                    for (int dx = 0; dx < cell_w; ++dx) {
                        sum += src_row[dx];
                    }
                }
                row[py * pw + px] = sum * inv_cell;
            }
        }
    }
    return 0;
}
