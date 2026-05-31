// SDX CUDA Kernel Headers
// Ultra-fast GPU acceleration for image generation operations

#ifndef SDX_KERNELS_H
#define SDX_KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// QUANTIZATION
// ============================================================================

void cuda_quantize_int8(
    float* input,
    int8_t* output,
    float scale,
    int size
);

void cuda_dequantize_int8(
    int8_t* input,
    float* output,
    float inv_scale,
    int size
);

void cuda_quantize_fp8(
    float* input,
    uint8_t* output,
    float scale,
    int size
);

// ============================================================================
// ACTIVATIONS
// ============================================================================

void cuda_relu(float* data, int size);

void cuda_gelu(float* input, float* output, int size);

void cuda_swish(float* input, float* output, int size);

void cuda_mish(float* input, float* output, int size);

// ============================================================================
// NORMALIZATION
// ============================================================================

void cuda_softmax(float* input, float* output, int seq_len);

void cuda_layer_norm(
    float* input,
    float* gamma,
    float* beta,
    float* output,
    int seq_len,
    int hidden_dim,
    float eps
);

void cuda_group_norm(
    float* input,
    float* gamma,
    float* beta,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int num_groups,
    float eps
);

void cuda_instance_norm(
    float* input,
    float* gamma,
    float* beta,
    float* output,
    int batch_size,
    int channels,
    int spatial_size,
    float eps
);

// ============================================================================
// LINEAR ALGEBRA
// ============================================================================

void cuda_matmul(
    float* a,
    float* b,
    float* c,
    int m,
    int n,
    int k
);

void cuda_batched_matmul(
    float* a,
    float* b,
    float* c,
    int batch_size,
    int m,
    int n,
    int k
);

void cuda_dot_product(
    float* a,
    float* b,
    float* result,
    int size
);

void cuda_cosine_similarity(
    float* a,
    float* b,
    float* result,
    int size
);

// ============================================================================
// ATTENTION
// ============================================================================

void cuda_attention(
    float* q,
    float* k,
    float* v,
    float* output,
    int seq_len,
    int hidden_dim,
    float scale
);

void cuda_flash_attention_v2(
    float* q,
    float* k,
    float* v,
    float* output,
    int seq_len,
    int hidden_dim,
    float scale
);

void cuda_grouped_query_attention(
    float* q,
    float* k,
    float* v,
    float* output,
    int seq_len,
    int hidden_dim,
    int num_groups,
    float scale
);

void cuda_rotary_embeddings(
    float* input,
    float* output,
    int seq_len,
    int hidden_dim,
    float base
);

// ============================================================================
// CONVOLUTION
// ============================================================================

void cuda_conv2d(
    float* input,
    float* kernel,
    float* output,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
);

void cuda_depthwise_conv2d(
    float* input,
    float* kernel,
    float* output,
    int channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding
);

// ============================================================================
// MEMORY AND UTILITY
// ============================================================================

void cuda_memcpy_host_to_device(void* host, void* device, size_t size);

void cuda_memcpy_device_to_host(void* device, void* host, size_t size);

void cuda_device_malloc(void** device_ptr, size_t size);

void cuda_device_free(void* device_ptr);

void cuda_synchronize();

#ifdef __cplusplus
}
#endif

#endif // SDX_KERNELS_H
