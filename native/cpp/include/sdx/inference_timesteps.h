/**
 * VP diffusion inference timestep path finalization (strictly decreasing indices).
 * C ABI for Python ctypes — optional acceleration for diffusion/inference_timesteps.py.
 */
#ifndef SDX_INFERENCE_TIMESTEPS_H
#define SDX_INFERENCE_TIMESTEPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(_WIN32) && defined(SDX_IT_BUILD)
#define SDX_IT_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_IT_API __declspec(dllimport)
#else
#define SDX_IT_API
#endif

/**
 * Match Python ``_resample_length`` + ``_enforce_strict_descending``:
 * clip, monotonize, resample/interpolate to ``target_len`` (best effort; result may be shorter
 * after deduplication, same as Python).
 *
 * @param raw Raw timestep indices (any order); length raw_n.
 * @param target_len Desired count (if <= 0, writes single 0).
 * @param num_train Training timesteps T (indices in [0, num_train-1]).
 * @param out Output buffer, capacity out_cap int64_t elements.
 * @return Number of elements written, or -1 on error (out_cap too small or invalid args).
 */
SDX_IT_API int64_t sdx_it_finalize_path(const int64_t *raw, int64_t raw_n, int target_len, int num_train,
                                        int64_t *out, int64_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* SDX_INFERENCE_TIMESTEPS_H */
