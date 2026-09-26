/*
 * Classic CFG combine — C ABI for Python ctypes.
 * Build (MSVC): cl /LD /O2 /I..\include /DSDX_C_CFG_COMBINE_BUILD /Fe:sdx_c_cfg_combine.dll sdx_c_cfg_combine.c
 * Build (gcc):  cc -O3 -shared -fPIC -I../include -DSDX_C_CFG_COMBINE_BUILD -o libsdx_c_cfg_combine.so sdx_c_cfg_combine.c
 */
#ifndef SDX_C_CFG_COMBINE_BUILD
#  define SDX_C_CFG_COMBINE_BUILD
#endif
#include "../include/sdx_c_cfg_combine.h"
#include <math.h>

int sdx_c_cfg_combine_f32(
    const float *cond,
    const float *uncond,
    float *out,
    size_t n,
    float scale,
    float rescale_phi
) {
    if (!cond || !uncond || !out || n == 0) {
        return -1;
    }

    double sum_c = 0.0, sum_c2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float v = uncond[i] + scale * (cond[i] - uncond[i]);
        out[i] = v;
        sum_c += (double)cond[i];
        sum_c2 += (double)cond[i] * (double)cond[i];
    }

    if (rescale_phi <= 0.0f) {
        return 0;
    }

    double mean_c = sum_c / (double)n;
    double var_c = sum_c2 / (double)n - mean_c * mean_c;
    if (var_c < 1e-12) {
        return 0;
    }
    double std_c = sqrt(var_c);

    double sum_o = 0.0, sum_o2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_o += (double)out[i];
        sum_o2 += (double)out[i] * (double)out[i];
    }
    double mean_o = sum_o / (double)n;
    double var_o = sum_o2 / (double)n - mean_o * mean_o;
    if (var_o < 1e-12) {
        return 0;
    }
    double std_o = sqrt(var_o);
    double ratio = std_c / std_o;
    float phi = rescale_phi;
    if (phi > 1.0f) {
        phi = 1.0f;
    }

    for (size_t i = 0; i < n; ++i) {
        float centered = (float)(((double)out[i] - mean_o) * ratio + mean_o);
        out[i] = phi * centered + (1.0f - phi) * out[i];
    }
    return 0;
}
