#include <cstdint>

#include "kernels.h"

__global__ void gpuMemoryReadKernel(uint8_t* __restrict memory, int n) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint8_t accum = 0;
    if (tid < n) {
        accum = memory[tid];
    }
}

void gpuMemoryRead(uint8_t* memory, int n) {
    constexpr int NUM_THREADS = 1024;
    const int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
    gpuMemoryReadKernel<<<NUM_BLOCKS, NUM_THREADS>>>(memory, n);
}

__global__ void gpuMemoryWriteKernel(uint8_t* __restrict memory, int n) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n) {
        memory[tid] = 42;
    }
}

void gpuMemoryWrite(uint8_t* memory, int n) {
    constexpr int NUM_THREADS = 1024;
    const int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;
    gpuMemoryWriteKernel<<<NUM_BLOCKS, NUM_THREADS>>>(memory, n);
}
