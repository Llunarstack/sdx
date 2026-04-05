/**
 * Optional VP beta schedule helpers (C ABI for Python ctypes).
 */
#ifndef SDX_BETA_SCHEDULES_H
#define SDX_BETA_SCHEDULES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(_WIN32) && defined(SDX_BS_BUILD)
#define SDX_BS_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_BS_API __declspec(dllimport)
#else
#define SDX_BS_API
#endif

/**
 * Squared-cosine v2 betas (Improved DDPM / diffusers style), same as
 * ``diffusion.schedules.squared_cosine_beta_schedule_v2`` + ``_clip_betas``.
 *
 * @param n Number of timesteps (>= 1).
 * @param max_beta Upper cap per step (typically 0.999).
 * @param out Output array, length >= n.
 * @param cap Capacity of out (must be >= n).
 * @return n on success, -1 on error.
 */
SDX_BS_API int64_t sdx_squared_cosine_betas_v2(int n, double max_beta, double *out, int64_t cap);

#ifdef __cplusplus
}
#endif

#endif /* SDX_BETA_SCHEDULES_H */
