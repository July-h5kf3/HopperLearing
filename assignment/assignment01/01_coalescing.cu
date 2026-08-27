// 实验 1: Coalescing 决定有效带宽
// 对应正文: HBM3 / L2 Cache (128B 缓存行 = 4x32B sector) / LSU 的 coalescing
//
//   nvcc -O3 -arch=native 01_coalescing.cu -o 01 && ./01
//
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(e));                                   \
            return 1;                                                        \
        }                                                                    \
    } while (0)

// warp 内 32 个线程地址连续: 合并成最少的 sector
__global__ void copy_coalesced(const float* __restrict__ in,
                               float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// warp 内 32 个线程落在 32 个不同的 128B 缓存行上:
// 每个被取上来的 32B sector 只有 4B 被真正用到
__global__ void copy_strided(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[(long long)i * 32 % n];
}

int main() {
    const int n = 1 << 26;                 // 256 MB, 远大于 50MB 的 L2
    float *in, *out;
    CHECK(cudaMalloc(&in, (size_t)n * 4));
    CHECK(cudaMalloc(&out, (size_t)n * 4));
    CHECK(cudaMemset(in, 1, (size_t)n * 4));

    const dim3 grid((n + 255) / 256);
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    const char* names[2] = {"coalesced", "strided "};
    for (int k = 0; k < 2; ++k) {
        if (k == 0) copy_coalesced<<<grid, 256>>>(in, out, n);   // 预热
        else        copy_strided <<<grid, 256>>>(in, out, n);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 10; ++r) {
            if (k == 0) copy_coalesced<<<grid, 256>>>(in, out, n);
            else        copy_strided <<<grid, 256>>>(in, out, n);
        }
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 10;
        double gb = 2.0 * n * 4 / 1e9;     // 读 + 写
        printf("%s  %8.3f ms   有效带宽 %7.1f GB/s\n", names[k], ms,
               gb / (ms / 1e3));
    }

    cudaFree(in);
    cudaFree(out);
    return 0;
}
