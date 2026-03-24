#include "sdx/hwc_to_chw.h"

#include <cuda_runtime.h>

__global__ void sdx_k_u8hwc3_to_chw_f32(const uint8_t *__restrict__ src, int H, int W,
                                       float *__restrict__ dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) {
        return;
    }
    const int idx = y * W * 3 + x * 3;
    const float r = static_cast<float>(src[idx]) * (1.0f / 255.0f);
    const float g = static_cast<float>(src[idx + 1]) * (1.0f / 255.0f);
    const float b = static_cast<float>(src[idx + 2]) * (1.0f / 255.0f);
    const int plane = H * W;
    dst[0 * plane + y * W + x] = r;
    dst[1 * plane + y * W + x] = g;
    dst[2 * plane + y * W + x] = b;
}

extern "C" int sdx_cuda_u8hwc3_to_chw_f32_host(const uint8_t *host_src, int H, int W, float *host_dst) {
    if (!host_src || !host_dst || H <= 0 || W <= 0) {
        return -2;
    }
    const size_t nbytes_src = static_cast<size_t>(H) * static_cast<size_t>(W) * 3u;
    const size_t nbytes_dst = static_cast<size_t>(3) * static_cast<size_t>(H) * static_cast<size_t>(W) * sizeof(float);

    uint8_t *d_src = nullptr;
    float *d_dst = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&d_src), nbytes_src);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&d_dst), nbytes_dst);
    if (e != cudaSuccess) {
        cudaFree(d_src);
        return -1;
    }
    e = cudaMemcpy(d_src, host_src, nbytes_src, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        return -1;
    }

    dim3 thr(16, 16);
    dim3 bl((static_cast<unsigned>(W) + 15u) / 16u, (static_cast<unsigned>(H) + 15u) / 16u);
    sdx_k_u8hwc3_to_chw_f32<<<bl, thr>>>(d_src, H, W, d_dst);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        return -1;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        return -1;
    }
    e = cudaMemcpy(host_dst, d_dst, nbytes_dst, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    return e == cudaSuccess ? 0 : -1;
}
