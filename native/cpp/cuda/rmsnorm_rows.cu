#include "sdx/rmsnorm_rows.h"

#include <cmath>
#include <cuda_runtime.h>

__global__ void sdx_k_rmsnorm_rows(const float *__restrict__ src, float *__restrict__ dst, int n_rows, int dim, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const float *r = src + static_cast<size_t>(row) * static_cast<size_t>(dim);
    float *out = dst + static_cast<size_t>(row) * static_cast<size_t>(dim);
    float s = 0.f;
    for (int i = 0; i < dim; ++i) {
        const float v = r[i];
        s += v * v;
    }
    const float inv_rms = rsqrtf(fmaxf(s / fmaxf(1, dim), eps));
    for (int i = 0; i < dim; ++i) {
        out[i] = r[i] * inv_rms;
    }
}

extern "C" int sdx_cuda_rmsnorm_rows_f32_host(float *host_data, int n_rows, int dim, float eps) {
    if (!host_data || n_rows <= 0 || dim <= 0) {
        return -2;
    }
    const size_t nbytes = static_cast<size_t>(n_rows) * static_cast<size_t>(dim) * sizeof(float);
    float *d = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&d), nbytes);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMemcpy(d, host_data, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(d);
        return -1;
    }
    const int threads = 256;
    const int blocks = (n_rows + threads - 1) / threads;
    const float eps_clamped = eps > 0.f ? eps : 1e-6f;
    sdx_k_rmsnorm_rows<<<blocks, threads>>>(d, d, n_rows, dim, eps_clamped);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(d);
        return -1;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cudaFree(d);
        return -1;
    }
    e = cudaMemcpy(host_data, d, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(d);
    return e == cudaSuccess ? 0 : -1;
}
