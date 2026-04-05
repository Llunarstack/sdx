#include "sdx/rmsnorm_rows_cpu.h"

#include <cmath>

extern "C" int sdx_rmsnorm_rows_f32_host(float *host_data, int n_rows, int dim, float eps) {
    if (!host_data || n_rows <= 0 || dim <= 0) {
        return -2;
    }
    const float eps_clamped = eps > 0.f ? eps : 1e-6f;
    for (int r = 0; r < n_rows; ++r) {
        float *row = host_data + static_cast<size_t>(r) * static_cast<size_t>(dim);
        float s = 0.f;
        for (int i = 0; i < dim; ++i) {
            const float v = row[i];
            s += v * v;
        }
        const float inv_rms = 1.0f / std::sqrt((s / static_cast<float>(dim)) + eps_clamped);
        for (int i = 0; i < dim; ++i) {
            row[i] *= inv_rms;
        }
    }
    return 0;
}
