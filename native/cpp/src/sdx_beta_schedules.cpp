/**
 * VP beta schedules - mirrors diffusion/schedules.py (squared cosine v2).
 */
#include "sdx/beta_schedules.h"

#include <cmath>
#include <cstdint>

static double clip_beta(double b) {
    if (b < 1e-4) {
        return 1e-4;
    }
    if (b > 0.999) {
        return 0.999;
    }
    return b;
}

extern "C" int64_t sdx_squared_cosine_betas_v2(int n, double max_beta, double *out, int64_t cap) {
    if (n < 1 || out == nullptr || cap < n || max_beta <= 0.0) {
        return -1;
    }
    const double scale = 1.008;
    const double off = 0.008;
    const double pi_h = 3.14159265358979323846 * 0.5;

    for (int i = 0; i < n; ++i) {
        const double t1 = static_cast<double>(i) / static_cast<double>(n);
        const double t2 = static_cast<double>(i + 1) / static_cast<double>(n);
        const double c1 = std::cos((t1 + off) / scale * pi_h);
        const double c2 = std::cos((t2 + off) / scale * pi_h);
        const double ab1 = c1 * c1;
        const double ab2 = c2 * c2;
        const double denom = ab1 > 1e-12 ? ab1 : 1e-12;
        double b = 1.0 - ab2 / denom;
        if (b > max_beta) {
            b = max_beta;
        }
        out[i] = clip_beta(b);
    }
    return static_cast<int64_t>(n);
}
