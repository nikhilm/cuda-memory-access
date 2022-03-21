#include <cstdint>
#include <iostream>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "kernels.h"

#define REPEAT2(x) x x
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT(x) REPEAT32(x)

// navcam images with only the Y plane are 780x780 = 608400 bytes, 912600 bytes including chroma.
// subject images can apparently be ~6mb.
#define ARGS ->RangeMultiplier(2)->Range(1 << 19, 1 << 23)
// Use manual timing so we can use the CUDA APIs to measure GPU time.
#define GPU_READ_ARGS ->RangeMultiplier(2)->Range(1 << 19, 1 << 23)->UseManualTime();

// Jetson Nano
constexpr std::size_t L1_CACHE_SIZE = 32 * 1024;
constexpr std::size_t L2_CACHE_SIZE = 2048 * 1024;

template <class Word>
static void BM_CpuPageableHostMemoryRead(benchmark::State& state) {
    const size_t size = state.range(0);
    void* memory = malloc(size);
    void* const end = static_cast<uint8_t*>(memory) + size;
    volatile Word* const p0 = static_cast<Word*>(memory);
    Word* const p1 = static_cast<Word*>(end);
    for (auto _ : state) {
        for (volatile Word* p = p0; p != p1; ) {
            REPEAT(benchmark::DoNotOptimize(*p++);)
        }
        benchmark::ClobberMemory();
    }
    free(memory);
    state.SetBytesProcessed(size * state.iterations());
    state.SetItemsProcessed((p1 - p0) * state.iterations());
}

// BENCHMARK_TEMPLATE1(BM_CpuPageableHostMemoryRead, float) ARGS;

template <class Word>
static void WriteToMemory(benchmark::State& state, void* memory, size_t size) {
    void* const end = static_cast<uint8_t*>(memory) + size;
    volatile Word* const p0 = static_cast<Word*>(memory);
    Word* const p1 = static_cast<Word*>(end);
    Word fill = {};
    for (auto _ : state) {
        for (volatile Word* p = p0; p != p1; ) {
            REPEAT(benchmark::DoNotOptimize(*p++ = fill);)
        }
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(size * state.iterations());
    state.SetItemsProcessed((p1 - p0) * state.iterations());
}

template <class Word>
static void BM_CpuPageableHostMemoryWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    void* memory = malloc(size);
    WriteToMemory<Word>(state, memory, size);
    free(memory);
}

BENCHMARK_TEMPLATE1(BM_CpuPageableHostMemoryWrite, uint8_t) ARGS;
BENCHMARK_TEMPLATE1(BM_CpuPageableHostMemoryWrite, float) ARGS;
BENCHMARK_TEMPLATE1(BM_CpuPageableHostMemoryWrite, double) ARGS;

template <class Word>
static void BM_PinnedHostMemoryWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    void* memory;
    if (cudaMallocHost(&memory, size) != cudaSuccess) {
        state.SkipWithError("cudaMallocHost failed");
        return;
    }
    WriteToMemory<Word>(state, memory, size);
    cudaFreeHost(memory);
}

BENCHMARK_TEMPLATE1(BM_PinnedHostMemoryWrite, uint8_t) ARGS;
BENCHMARK_TEMPLATE1(BM_PinnedHostMemoryWrite, float) ARGS;
BENCHMARK_TEMPLATE1(BM_PinnedHostMemoryWrite, double) ARGS;

template <class Word>
static void BM_PinnedMappedHostMemoryWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    void* memory;
    if (cudaHostAlloc(&memory, size, cudaHostAllocMapped) != cudaSuccess) {
        state.SkipWithError("cudaHostAlloc failed");
        return;
    }
    WriteToMemory<Word>(state, memory, size);
    cudaFreeHost(memory);
}

BENCHMARK_TEMPLATE1(BM_PinnedMappedHostMemoryWrite, uint8_t) ARGS;
BENCHMARK_TEMPLATE1(BM_PinnedMappedHostMemoryWrite, float) ARGS;
BENCHMARK_TEMPLATE1(BM_PinnedMappedHostMemoryWrite, double) ARGS;

template <class Word>
static void BM_UnifiedMemoryWrite(benchmark::State& state) {
    const size_t size = state.range(0);
    void* memory;
    if (cudaMallocManaged(&memory, size, cudaHostAllocMapped) != cudaSuccess) {
        state.SkipWithError("cudaHostAlloc failed");
        return;
    }
    WriteToMemory<Word>(state, memory, size);
    cudaFreeHost(memory);
}

BENCHMARK_TEMPLATE1(BM_UnifiedMemoryWrite, uint8_t) ARGS;
BENCHMARK_TEMPLATE1(BM_UnifiedMemoryWrite, float) ARGS;
BENCHMARK_TEMPLATE1(BM_UnifiedMemoryWrite, double) ARGS;

static void BM_PageableHostToGPUCopy(benchmark::State& state) {
    const size_t size = state.range(0);
    void* memory = malloc(size);
    void* dst;
    if (cudaMalloc(&dst, size) != cudaSuccess) {
        state.SkipWithError("Device memory allocation failed");
        free(memory);
        return;
    }

    for (auto _ : state) {
        if (cudaMemcpy(dst, memory, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpy failed");
            break;
        }
        benchmark::ClobberMemory();
    }
    free(memory);
    cudaFree(dst);
    state.SetBytesProcessed(size * state.iterations());
}

BENCHMARK(BM_PageableHostToGPUCopy)->RangeMultiplier(2)->Range(1<<20, 1<<24);

static void BM_PinnedHostToGPUCopy(benchmark::State& state) {
    const size_t size = state.range(0);
    void* memory;
    if (cudaMallocHost(&memory, size) != cudaSuccess) {
        state.SkipWithError("Host memory allocation failed");
        return;
    }

    void* dst;
    if (cudaMalloc(&dst, size) != cudaSuccess) {
        state.SkipWithError("Device memory allocation failed");
        free(memory);
        return;
    }

    for (auto _ : state) {
        if (cudaMemcpy(dst, memory, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            state.SkipWithError("cudaMemcpy failed");
            break;
        }
        benchmark::ClobberMemory();
    }
    cudaFreeHost(memory);
    cudaFree(dst);
    state.SetBytesProcessed(size * state.iterations());
}

BENCHMARK(BM_PinnedHostToGPUCopy)->RangeMultiplier(2)->Range(1<<20, 1<<24);

static void BM_GPUMemoryRead(benchmark::State& state) {
    const size_t size = state.range(0);
    // Allocate GPU memory directly.
    uint8_t* gpu_memory;
    if (cudaMalloc(&gpu_memory, size) != cudaSuccess) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }

    uint8_t* memory = static_cast<uint8_t*>(calloc(size, sizeof(uint8_t)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
        
    for (auto _ : state) {
        // Do a memcpy to force cache invalidation on the GPU side.
        if (cudaMemcpy(gpu_memory, static_cast<void*>(memory), size, cudaMemcpyHostToDevice) != cudaSuccess) {
            state.SkipWithError("DeviceToHost copy failed");
        }

        cudaEventRecord(start);
        gpuMemoryRead(gpu_memory, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }

    cudaFree(gpu_memory);
    free(memory);
    state.SetBytesProcessed(size * state.iterations());
}

BENCHMARK(BM_GPUMemoryRead) GPU_READ_ARGS;

static void BM_GPUUnifiedMemoryRead(benchmark::State& state) {
    const size_t size = state.range(0);
    // Allocate unified memory.
    uint8_t* gpu_memory;
    if (cudaMallocManaged(&gpu_memory, size) != cudaSuccess) {
        state.SkipWithError("cudaMallocManaged failed");
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
        
    for (auto _ : state) {
        // Invalidate unified memory by writing to it from the host.
        for (size_t i = 0; i < size; ++i) {
            gpu_memory[i] = 0;
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        gpuMemoryRead(gpu_memory, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }

    cudaDeviceSynchronize();
    cudaFree(gpu_memory);
    state.SetBytesProcessed(size * state.iterations());
}

BENCHMARK(BM_GPUUnifiedMemoryRead) GPU_READ_ARGS;

static void BM_GPUUnifiedMemoryPrefetchRead(benchmark::State& state) {
    const size_t size = state.range(0);
    // Allocate unified memory.
    uint8_t* gpu_memory;
    if (cudaMallocManaged(&gpu_memory, size) != cudaSuccess) {
        state.SkipWithError("cudaMallocManaged failed");
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        state.SkipWithError("cudaGetDevice failed");
    }

    for (auto _ : state) {
        // Invalidate unified memory by writing to it from the host.
        for (size_t i = 0; i < size; ++i) {
            gpu_memory[i] = 0;
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        if (cudaStreamAttachMemAsync(0, gpu_memory, size, cudaMemAttachGlobal) != cudaSuccess) {
            state.SkipWithError("Prefetch failed");
            break;
        }
        gpuMemoryRead(gpu_memory, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        state.SetIterationTime(elapsed_ms / 1000.0);
    }

    cudaDeviceSynchronize();
    cudaFree(gpu_memory);
    state.SetBytesProcessed(size * state.iterations());
}

BENCHMARK(BM_GPUUnifiedMemoryPrefetchRead) GPU_READ_ARGS;

BENCHMARK_MAIN();
