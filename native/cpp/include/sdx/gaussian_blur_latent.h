/**
 * CUDA depthwise Gaussian blur on a float32 latent (B, C, H, W).
 * Matches gaussian_blur_latent() in diffusion/sampling_utils.py and
 * _gaussian_blur() in diffusion/holy_grail/latent_refiner.py.
 */
#ifndef SDX_GAUSSIAN_BLUR_LATENT_H
#define SDX_GAUSSIAN_BLUR_LATENT_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_BLUR_BUILD)
#  define SDX_CUDA_BLUR_API __declspec(dllexport)
#elif defined(_WIN32)
#  define SDX_CUDA_BLUR_API __declspec(dllimport)
#else
#  define SDX_CUDA_BLUR_API
#endif

/**
 * Blur src_host (B*C*H*W floats, row-major BCHW) into dst_host with a
 * Gaussian kernel of the given sigma. Radius = clamp(round(sigma*2), 1, 7).
 *
 * src_host and dst_host may be the same pointer (in-place).
 * @return 0 on success, -1 on CUDA error, -2 on bad arguments.
 */
SDX_CUDA_BLUR_API int sdx_cuda_gaussian_blur_latent_f32(
    const float *src_host,
    float       *dst_host,
    int B, int C, int H, int W,
    float sigma);

#ifdef __cplusplus
}
#endif

#endif /* SDX_GAUSSIAN_BLUR_LATENT_H */
