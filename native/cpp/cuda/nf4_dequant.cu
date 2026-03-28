#include "sdx/nf4_dequant.h"

#include <cuda_runtime.h>

__constant__ float k_nf4_table[16] = {
    -1.0f,         -0.6961928f,   -0.52507305f, -0.39491749f, -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
    0.07993701f,   0.16089617f,   0.24675329f,  0.33791524f,  0.44070983f,  0.562617f,    0.72295684f,  1.0f,
};

__global__ void sdx_k_nf4_dequant(const uint8_t *__restrict__ packed, const float *__restrict__ absmax,
                                  float *__restrict__ out, int n_blocks, int block_size) {
    const int bi = blockIdx.x;
    const int ti = threadIdx.x;
    if (bi >= n_blocks) {
        return;
    }
    const int base = bi * block_size;
    const float am = absmax[bi];
    for (int j = ti; j < block_size; j += blockDim.x) {
        const int g = base + j;
        const int byte_idx = g / 2;
        const uint8_t b = packed[byte_idx];
        const int code = (g % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
        out[g] = k_nf4_table[code] * am;
    }
}

extern "C" int sdx_cuda_nf4_dequant_f32_host(const uint8_t *packed, const float *absmax, float *out,
                                             int32_t n_blocks, int32_t block_size, int32_t n_weights) {
    if (!packed || !absmax || !out || n_blocks <= 0 || block_size <= 0 || n_weights <= 0) {
        return -2;
    }
    if (n_weights != n_blocks * block_size) {
        return -2;
    }
    const int nbytes_packed = (n_weights + 1) / 2;
    uint8_t *dp = nullptr;
    float *da = nullptr;
    float *dout = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&dp), static_cast<size_t>(nbytes_packed));
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&da), static_cast<size_t>(n_blocks) * sizeof(float));
    if (e != cudaSuccess) {
        cudaFree(dp);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&dout), static_cast<size_t>(n_weights) * sizeof(float));
    if (e != cudaSuccess) {
        cudaFree(dp);
        cudaFree(da);
        return -1;
    }
    e = cudaMemcpy(dp, packed, static_cast<size_t>(nbytes_packed), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(da, absmax, static_cast<size_t>(n_blocks) * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    sdx_k_nf4_dequant<<<n_blocks, 128>>>(dp, da, dout, n_blocks, block_size);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(out, dout, static_cast<size_t>(n_weights) * sizeof(float), cudaMemcpyDeviceToHost);
fail:
    cudaFree(dp);
    cudaFree(da);
    cudaFree(dout);
    return e == cudaSuccess ? 0 : -1;
}
