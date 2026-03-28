#include "sdx/silu_gate.h"

#include <cuda_runtime.h>

__global__ void sdx_k_silu_gate(const float *__restrict__ x, const float *__restrict__ g, float *__restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float v = x[i];
    const float sig = 1.f / (1.f + expf(-v));
    out[i] = (v * sig) * g[i];
}

extern "C" int sdx_cuda_silu_gate_f32_host(
    const float *x_host,
    const float *gate_host,
    float *out_host,
    int64_t n) {
    if (!x_host || !gate_host || !out_host || n <= 0) {
        return -2;
    }
    const size_t nbytes = static_cast<size_t>(n) * sizeof(float);
    float *x_dev = nullptr;
    float *g_dev = nullptr;
    float *o_dev = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&x_dev), nbytes);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&g_dev), nbytes);
    if (e != cudaSuccess) {
        cudaFree(x_dev);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&o_dev), nbytes);
    if (e != cudaSuccess) {
        cudaFree(x_dev);
        cudaFree(g_dev);
        return -1;
    }
    e = cudaMemcpy(x_dev, x_host, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(x_dev);
        cudaFree(g_dev);
        cudaFree(o_dev);
        return -1;
    }
    e = cudaMemcpy(g_dev, gate_host, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(x_dev);
        cudaFree(g_dev);
        cudaFree(o_dev);
        return -1;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    sdx_k_silu_gate<<<blocks, threads>>>(x_dev, g_dev, o_dev, n);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cudaFree(x_dev);
        cudaFree(g_dev);
        cudaFree(o_dev);
        return -1;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cudaFree(x_dev);
        cudaFree(g_dev);
        cudaFree(o_dev);
        return -1;
    }
    e = cudaMemcpy(out_host, o_dev, nbytes, cudaMemcpyDeviceToHost);
    cudaFree(x_dev);
    cudaFree(g_dev);
    cudaFree(o_dev);
    return e == cudaSuccess ? 0 : -1;
}
