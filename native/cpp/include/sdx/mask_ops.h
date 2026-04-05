/**
 * CPU mask → patch-weight helpers for part-aware training.
 * Matches image_mask_to_patch_weights() in utils/training/part_aware_training.py.
 */
#ifndef SDX_MASK_OPS_H
#define SDX_MASK_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_MASK_OPS_BUILD)
#  define SDX_MASK_OPS_API __declspec(dllexport)
#elif defined(_WIN32)
#  define SDX_MASK_OPS_API __declspec(dllimport)
#else
#  define SDX_MASK_OPS_API
#endif

/**
 * Downsample (B, 1, H, W) float32 mask to (B, ph*pw) via average pooling.
 *
 * H must be divisible by ph; W must be divisible by pw.
 * out must be pre-allocated to B * ph * pw floats.
 *
 * @return 0 on success, -2 on bad args, -3 if H%ph != 0 or W%pw != 0.
 */
SDX_MASK_OPS_API int sdx_mask_to_patch_weights_f32(
    const float *mask,
    float       *out,
    int B, int H, int W,
    int ph, int pw);

#ifdef __cplusplus
}
#endif

#endif /* SDX_MASK_OPS_H */
