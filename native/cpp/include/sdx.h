/*
 * SDX native C ABI (loaded from Python with ctypes by native/_experimental/python/sdx_native/ffi.py).
 *   sdx_cpu  : latent geometry, inference-timestep finalization, beta schedule, image metrics, mask pooling.
 *   sdx_cuda : the sdx_cuda_* functions (optional, -DSDX_BUILD_CUDA=ON).
 * Unless noted, functions return 0 on success and a negative code on bad arguments / CUDA errors.
 */
#ifndef SDX_H
#define SDX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_BUILD)
#define SDX_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_API __declspec(dllimport)
#else
#define SDX_API __attribute__((visibility("default")))
#endif

/* ---- latent geometry: each returns 0 when inputs are non-positive or not divisible ---- */
SDX_API int sdx_latent_spatial_size(int image_hw, int vae_scale);
SDX_API int sdx_patch_grid_dim(int latent_hw, int patch_size);
SDX_API int sdx_num_patch_tokens(int image_hw, int vae_scale, int patch_size);
SDX_API int sdx_latent_numel(int channels, int latent_h, int latent_w);

/*
 * Mirrors diffusion/inference_timesteps.py (_enforce_strict_descending + _resample_length_numpy).
 * Returns the number of int64 indices written to out, or -1 on error / out_cap too small.
 */
SDX_API int64_t sdx_it_finalize_path(const int64_t *raw, int64_t raw_n, int target_len, int num_train,
                                     int64_t *out, int64_t out_cap);

/* Squared-cosine v2 betas, clipped to [1e-4, 0.999] and capped at max_beta. Returns n, or -1. */
SDX_API int64_t sdx_squared_cosine_betas_v2(int n, double max_beta, double *out, int64_t cap);

/* ---- uint8 HWC image metrics (luma = (77R + 150G + 29B + 128) >> 8) ---- */
SDX_API int sdx_image_mean_luma_u8(const unsigned char *hwc, int height, int width, int channels,
                                   double *out_mean_luma);
SDX_API int sdx_image_clip_ratio_u8(const unsigned char *hwc, int height, int width, int channels,
                                    unsigned char clip_low, unsigned char clip_high, double *out_ratio);
SDX_API int sdx_image_laplacian_var_u8(const unsigned char *hwc, int height, int width, int channels,
                                       double *out_lap_var);
/* Count of 4-connected components with luma <= threshold and min_area <= area <= max_area (0 = unbounded). */
SDX_API int sdx_image_count_components_u8(const unsigned char *hwc, int height, int width, int channels,
                                          unsigned char threshold, int min_area, int max_area);

/* (B, 1, H, W) mask -> (B, ph*pw) cell means. -2 bad args, -3 if H % ph or W % pw != 0. */
SDX_API int sdx_mask_to_patch_weights_f32(const float *mask, float *out, int B, int H, int W, int ph, int pw);

/* ---- CUDA (host pointers in/out) ---- */
SDX_API int sdx_cuda_gaussian_blur_latent_f32(const float *src_host, float *dst_host, int B, int C, int H, int W,
                                              float sigma);
SDX_API int sdx_cuda_luma_stats_u8_host(const uint8_t *host_src, int height, int width, int channels,
                                        uint8_t clip_low, uint8_t clip_high, float *out_mean_luma,
                                        float *out_clip_ratio);
SDX_API int sdx_cuda_l2_normalize_rows_f32_host(float *host_data, int n_rows, int dim, float eps);
/* Index and dot product of the candidate row with the largest dot against query (first wins ties). */
SDX_API int sdx_cuda_style_pick_best_f32_host(const float *query, const float *candidates, int n_cand, int dim,
                                              int *out_index, float *out_score);

#ifdef __cplusplus
}
#endif

#endif /* SDX_H */
