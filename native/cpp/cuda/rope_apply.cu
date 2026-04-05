#include "sdx/rope_apply.h"

#include <cmath>
#include <cuda_runtime.h>

__global__ void sdx_k_apply_rope_interleaved(float *q, float *k, int n_tokens, int dim, float theta_base) {
    const int half = dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // token*half + pair
    const int total = n_tokens * half;
    if (idx >= total) {
        return;
    }
    const int t = idx / half;
    const int p = idx % half;
    const int d0 = p * 2;
    const int d1 = d0 + 1;
    const float inv_freq = powf(theta_base, -2.0f * (static_cast<float>(p) / static_cast<float>(dim)));
    const float ang = static_cast<float>(t) * inv_freq;
    const float c = cosf(ang);
    const float s = sinf(ang);
    const size_t off = static_cast<size_t>(t) * static_cast<size_t>(dim);

    const float q0 = q[off + d0];
    const float q1 = q[off + d1];
    q[off + d0] = q0 * c - q1 * s;
    q[off + d1] = q0 * s + q1 * c;

    const float k0 = k[off + d0];
    const float k1 = k[off + d1];
    k[off + d0] = k0 * c - k1 * s;
    k[off + d1] = k0 * s + k1 * c;
}

extern "C" int sdx_cuda_apply_rope_interleaved_f32_host(
    float *q_host,
    float *k_host,
    int n_tokens,
    int dim,
    float theta_base) {
    if (!q_host || !k_host || n_tokens <= 0 || dim <= 0 || (dim % 2) != 0) {
        return -2;
    }
    const size_t n = static_cast<size_t>(n_tokens) * static_cast<size_t>(dim);
    const size_t nbytes = n * sizeof(float);
    float *q_dev = nullptr;
    float *k_dev = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&q_dev), nbytes);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&k_dev), nbytes);
    if (e != cudaSuccess) {
        cudaFree(q_dev);
        return -1;
    }
    e = cudaMemcpy(q_dev, q_host, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(q_dev);
        cudaFree(k_dev);
        return -1;
    }
    e = cudaMemcpy(k_dev, k_host, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(q_dev);
        cudaFree(k_dev);
        return -1;
    }
    const int half = dim / 2;
    const int total = n_tokens * half;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const float base = theta_base > 1.f ? theta_base : 10000.f;
    sdx_k_apply_rope_interleaved<<<blocks, threads>>>(q_dev, k_dev, n_tokens, dim, base);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(q_dev);
        cudaFree(k_dev);
        return -1;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cudaFree(q_dev);
        cudaFree(k_dev);
        return -1;
    }
    e = cudaMemcpy(q_host, q_dev, nbytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cudaFree(q_dev);
        cudaFree(k_dev);
        return -1;
    }
    e = cudaMemcpy(k_host, k_dev, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(q_dev);
    cudaFree(k_dev);
    return e == cudaSuccess ? 0 : -1;
}
