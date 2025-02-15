#include <iostream>
#include <cuda_runtime.h>
#include "kernel.h"

static constexpr size_t N = 256;

#define CHECKERR(EXPRESSION) do { \
        cudaError_t ret = (EXPRESSION); \
        if (ret != cudaError::cudaSuccess) { \
            std::cerr << "[CUDA Error] Assert `" << #EXPRESSION << "` failed. Cuda return: " << cudaGetErrorString(ret) << std::endl; \
            __debugbreak(); \
        } \
    } while (0);

__global__ void __kernel__vectorAdd(const float *a, const float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void vectorAdd(const float *a, const float *b, float *c) {
    printf("kernel started.\n");

    static constexpr size_t size = sizeof(float) * N;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    CHECKERR(cudaMemcpy(d_a, a, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CHECKERR(cudaMemcpy(d_b, b, size, cudaMemcpyKind::cudaMemcpyHostToDevice));

    __kernel__vectorAdd<<< 1, N >>>(d_a, d_b, d_c);
    CHECKERR(cudaDeviceSynchronize());

    CHECKERR(cudaMemcpy(c, d_c, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    CHECKERR(cudaFree(d_a));
    CHECKERR(cudaFree(d_b));
    CHECKERR(cudaFree(d_c));

}

#include <Windows.h>

void *getCuMemAddress(HANDLE handle, size_t size) {
    cudaExternalMemory_t externalMemory;
    cudaExternalMemoryHandleDesc desc;

    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleType::cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;

    CHECKERR(cudaImportExternalMemory(&externalMemory, &desc));

    cudaExternalMemoryBufferDesc bufferDesc;

    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = size;
    bufferDesc.flags = 0;

    void *buf;
    CHECKERR(cudaExternalMemoryGetMappedBuffer(&buf, externalMemory, &bufferDesc));
    
    return buf;
}

void vectorAdd(HANDLE a, HANDLE b, HANDLE c) {
    printf("kernel (external memory) started!\n");
    static constexpr size_t size = sizeof(float) * N;

    float *d_a = static_cast<float*>(getCuMemAddress(a, size));
    float *d_b = static_cast<float *>(getCuMemAddress(b, size));
    float *d_c = static_cast<float *>(getCuMemAddress(c, size));

    dim3 gridSize{ 1 };
    dim3 blockSize{ N };
    __kernel__vectorAdd << < gridSize, blockSize >> > (d_a, d_b, d_c);
    CHECKERR(cudaDeviceSynchronize());

}
