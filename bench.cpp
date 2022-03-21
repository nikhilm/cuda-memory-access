#include <cstdint>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#define REPEAT2(x) x x
#define REPEAT4(x) REPEAT2(x) REPEAT2(x)
#define REPEAT8(x) REPEAT4(x) REPEAT4(x)
#define REPEAT16(x) REPEAT8(x) REPEAT8(x)
#define REPEAT32(x) REPEAT16(x) REPEAT16(x)
#define REPEAT(x) REPEAT32(x)

#define ARGS ->RangeMultiplier(2)->Range(1<<10, 1<<20)

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

// BENCHMARK_TEMPLATE1(BM_CpuPageableHostMemoryRead, float)->RangeMultiplier(2)->Range(1<<10, 1<<20);

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
}

BENCHMARK(BM_PinnedHostToGPUCopy)->RangeMultiplier(2)->Range(1<<20, 1<<24);

BENCHMARK_MAIN();
