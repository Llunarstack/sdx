/**
 * Optional CUDA: uint8 NHWC (H,W,3) -> float32 NCHW (3,H,W), values scaled to [0,1].
 * Built only when CMake -DSDX_BUILD_CUDA=ON (requires nvcc). C ABI for experiments / ctypes.
 */
#ifndef SDX_HWC_TO_CHW_H
#define SDX_HWC_TO_CHW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_CUDA_HWC_BUILD)
#define SDX_CUDA_HWC_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_HWC_API __declspec(dllimport)
#else
#define SDX_CUDA_HWC_API
#endif

/**
 * Host pointers: allocates device memory, runs kernel, copies back. For benchmarking / small batches.
 * @return 0 on success, non-zero on CUDA error
 */
SDX_CUDA_HWC_API int sdx_cuda_u8hwc3_to_chw_f32_host(const uint8_t *host_src, int H, int W, float *host_dst);

#ifdef __cplusplus
}
#endif

#endif /* SDX_HWC_TO_CHW_H */
