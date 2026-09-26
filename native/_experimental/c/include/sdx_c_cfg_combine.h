#ifndef SDX_C_CFG_COMBINE_H
#define SDX_C_CFG_COMBINE_H

#include <stddef.h>

#if defined(_WIN32) || defined(_WIN64)
#  ifdef SDX_C_CFG_COMBINE_BUILD
#    define SDX_C_CFG_API __declspec(dllexport)
#  else
#    define SDX_C_CFG_API __declspec(dllimport)
#  endif
#else
#  define SDX_C_CFG_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Classic CFG: out = uncond + scale * (cond - uncond), optional std rescale.
 * All buffers are float32, length `n`. Returns 0 on success, -1 on bad args.
 *
 * If rescale_phi > 0, blend toward cond's std: out = phi*out_norm + (1-phi)*out
 * where out_norm matches std(cond) (CFG rescale heuristic).
 */
SDX_C_CFG_API int sdx_c_cfg_combine_f32(
    const float *cond,
    const float *uncond,
    float *out,
    size_t n,
    float scale,
    float rescale_phi
);

#ifdef __cplusplus
}
#endif

#endif /* SDX_C_CFG_COMBINE_H */
