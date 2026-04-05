#include "sdx/flow_matching_velocity.h"

#include <cuda_runtime.h>

__global__ void sdx_k_flow_velocity_residual(const float *__restrict__ x0, const float *__restrict__ eps,
                                             float *__restrict__ out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (i < n) {
        out[i] = eps[i] - x0[i];
    }
}

extern "C" int sdx_cuda_flow_velocity_residual_f32_host(const float *x0, const float *eps, float *out, int64_t n) {
    if (!x0 || !eps || !out || n <= 0) {
        return -2;
    }
    const size_t nn = static_cast<size_t>(n);
    const size_t nbytes = nn * sizeof(float);
    float *dx0 = nullptr;
    float *deps = nullptr;
    float *dout = nullptr;
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&dx0), nbytes);
    if (e != cudaSuccess) {
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&deps), nbytes);
    if (e != cudaSuccess) {
        cudaFree(dx0);
        return -1;
    }
    e = cudaMalloc(reinterpret_cast<void **>(&dout), nbytes);
    if (e != cudaSuccess) {
        cudaFree(dx0);
        cudaFree(deps);
        return -1;
    }
    e = cudaMemcpy(dx0, x0, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(deps, eps, nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        goto fail;
    }
    {
        const int threads = 256;
        const int blocks = static_cast<int>((nn + threads - 1) / threads);
        sdx_k_flow_velocity_residual<<<blocks, threads>>>(dx0, deps, dout, nn);
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        goto fail;
    }
    e = cudaMemcpy(out, dout, nbytes, cudaMemcpyDeviceToHost);
fail:
    cudaFree(dx0);
    cudaFree(deps);
    cudaFree(dout);
    return e == cudaSuccess ? 0 : -1;
}
