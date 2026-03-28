/**
 * NF4 block dequantization (float32). Table matches ``utils/quantization/nf4_codec.py``.
 * Packed: two 4-bit indices per byte (low nibble first code, high nibble second).
 */
#ifndef SDX_NF4_DEQUANT_H
#define SDX_NF4_DEQUANT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(_WIN32) && defined(SDX_CUDA_NF4_BUILD)
#define SDX_CUDA_NF4_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_CUDA_NF4_API __declspec(dllimport)
#else
#define SDX_CUDA_NF4_API
#endif

/**
 * Host buffers: ``packed`` length = ceil(n_weights / 2), ``absmax`` length = n_blocks,
 * ``out`` length = n_weights (only first n_weights dequantized; pad was applied at quant time).
 */
SDX_CUDA_NF4_API int sdx_cuda_nf4_dequant_f32_host(const uint8_t *packed, const float *absmax, float *out,
                                                    int32_t n_blocks, int32_t block_size, int32_t n_weights);

#ifdef __cplusplus
}
#endif

#endif
