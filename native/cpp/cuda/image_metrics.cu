#include "sdx/image_metrics_cuda.h"

#include <cuda_runtime.h>

__global__ void sdx_k_luma_stats_u8(
    const uint8_t *__restrict__ src,
    int n_pixels,
    int channels,
    uint8_t clip_low,
    uint8_t clip_high,
    unsigned long long *__restrict__ out_sum,
    unsigned long long *__restrict__ out_clipped
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) {
        return;
    }
    const uint8_t *px = src + static_cast<size_t>(idx) * static_cast<size_t>(channels);
    uint8_t y;
    if (channels <= 1) {
        y = px[0];
    } else {
        const uint32_t r = static_cast<uint32_t>(px[0]);
        const uint32_t g = static_cast<uint32_t>(px[1]);
        const uint32_t b = static_cast<uint32_t>(px[2]);
        y = static_cast<uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
    }
    atomicAdd(out_sum, static_cast<unsigned long long>(y));
    if (y <= clip_low || y >= clip_high) {
        atomicAdd(out_clipped, 1ull);
    }
}

extern "C" int sdx_cuda_luma_stats_u8_host(
    const uint8_t *host_src,
    int height,
    int width,
    int channels,
    uint8_t clip_low,
    uint8_t clip_high,
    float *out_mean_luma,
    float *out_clip_ratio
) {
    if (!host_src || !out_mean_luma || !out_clip_ratio || height <= 0 || width <= 0 || channels <= 0) {
        return -2;
    }
    const int n_pixels = height * width;
    const size_t src_bytes = static_cast<size_t>(n_pixels) * static_cast<size_t>(channels);

    uint8_t *d_src = nullptr;
    unsigned long long *d_sum = nullptr;
    unsigned long long *d_clipped = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&d_src), src_bytes);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&d_sum), sizeof(unsigned long long));
    if (e != cudaSuccess) {
        cudaFree(d_src);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&d_clipped), sizeof(unsigned long long));
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        return -1;
    }
    e = cudaMemcpy(d_src, host_src, src_bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        cudaFree(d_clipped);
        return -1;
    }
    e = cudaMemset(d_sum, 0, sizeof(unsigned long long));
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        cudaFree(d_clipped);
        return -1;
    }
    e = cudaMemset(d_clipped, 0, sizeof(unsigned long long));
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        cudaFree(d_clipped);
        return -1;
    }

    const int threads = 256;
    const int blocks = (n_pixels + threads - 1) / threads;
    sdx_k_luma_stats_u8<<<blocks, threads>>>(d_src, n_pixels, channels, clip_low, clip_high, d_sum, d_clipped);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        cudaFree(d_clipped);
        return -1;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        cudaFree(d_clipped);
        return -1;
    }

    unsigned long long host_sum = 0ull;
    unsigned long long host_clipped = 0ull;
    e = cudaMemcpy(&host_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        cudaFree(d_clipped);
        return -1;
    }
    e = cudaMemcpy(&host_clipped, d_clipped, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_sum);
        cudaFree(d_clipped);
        return -1;
    }

    cudaFree(d_src);
    cudaFree(d_sum);
    cudaFree(d_clipped);

    const float denom = static_cast<float>(n_pixels > 0 ? n_pixels : 1);
    *out_mean_luma = static_cast<float>(static_cast<double>(host_sum) / static_cast<double>(denom));
    *out_clip_ratio = static_cast<float>(static_cast<double>(host_clipped) / static_cast<double>(denom));
    return 0;
}
