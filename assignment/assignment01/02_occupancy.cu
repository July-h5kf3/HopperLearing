// 实验 2: Occupancy 与 Latency Hiding
// 对应正文: Warp / Warp Scheduler / "为什么要让尽可能多的 warp 常驻"
//
//   nvcc -O3 -arch=native 02_occupancy.cu -o 02 && ./02
//
// 总工作量固定为 N 的拷贝, 用 #pragma unroll 1 保证每个 warp 同一时刻
// 只有一个 load 在飞 (store 依赖 load 的结果)。此时:
//   在飞请求数 ~= 常驻 warp 数
//   有效带宽  ~= 常驻 warp 数 x 每请求字节数 / 内存延迟   (Little's Law)
// 逐渐增大 grid (即提高 occupancy), 带宽应先线性上升, 饱和后变平。
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

__global__ void copy_kernel(const float* __restrict__ in,
                            float* __restrict__ out, long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
#pragma unroll 1
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i] = in[i];
    }
}

int main() {
    int sm_count = 0;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    const long long n = 1ll << 26;         // 256 MB in + 256 MB out
    float *in, *out;
    CHECK(cudaMalloc(&in, (size_t)n * 4));
    CHECK(cudaMalloc(&out, (size_t)n * 4));
    CHECK(cudaMemset(in, 1, (size_t)n * 4));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    printf("SM 数: %d, 总数据量: 读 256MB + 写 256MB\n\n", sm_count);
    printf("每 SM 常驻 block 数   grid       耗时       有效带宽\n");
    for (int bps = 1; bps <= 32; bps *= 2) {
        const int grid = sm_count * bps;
        copy_kernel<<<grid, 256>>>(in, out, n);                // 预热
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 5; ++r)
            copy_kernel<<<grid, 256>>>(in, out, n);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 5;
        double gb = 2.0 * n * 4 / 1e9;
        printf("%13d      %7d   %8.3f ms   %8.1f GB/s\n", bps, grid, ms,
               gb / (ms / 1e3));
    }

    printf("\n观察: 带宽先随常驻 warp 数近似线性增长, 达到峰值后变平 --\n");
    printf("变平的起点, 就是内存延迟被完全隐藏的位置。\n");

    cudaFree(in);
    cudaFree(out);
    return 0;
}
