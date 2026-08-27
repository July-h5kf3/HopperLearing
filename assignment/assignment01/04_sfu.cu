// 实验 4: SFU 吞吐瓶颈
// 对应正文: 每个 SM 只有 16 个 SFU, 而 FP32 Core 有 128 个 (吞吐比 8:1)
//
//   nvcc -O3 -arch=native 04_sfu.cu -o 04 && ./04
//
// 两个 kernel 循环次数完全相同, 唯一区别是循环体:
//   fmaf  -> 走 FP32 流水线 (每 SMSP 每 cycle 完成一个 warp 级 FMA)
//   __sinf -> 每次迭代发射一条 MUFU 指令, 走 SFU (每 SMSP 每 cycle 只完成 4 个线程)
//
// 注意: 必须用 __sinf intrinsic。普通的 sinf() 默认会编译成软件多项式
// (一堆 FMA), 那样测的还是 FP32 流水线, 根本碰不到 SFU。
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

__global__ void fma_kernel(const float* __restrict__ in,
                           float* __restrict__ out, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = in[i], c = 1e-6f;
    const float b = 1.0000001f;
    for (int r = 0; r < reps; ++r) c = fmaf(a, b, c);   // FP32 流水线
    out[i] = c;
}

__global__ void sfu_kernel(const float* __restrict__ in,
                           float* __restrict__ out, int reps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = in[i], c = 1e-6f;
    for (int r = 0; r < reps; ++r) c = __sinf(a + c);   // MUFU, 走 SFU
    out[i] = c;
}

int main() {
    int sm_count = 0;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    const int reps = 4096;
    const dim3 grid(sm_count * 8), block(256);         // 拉满 occupancy
    const long long total = (long long)grid.x * block.x * reps;

    float *in, *out;
    CHECK(cudaMalloc(&in, grid.x * block.x * 4));
    CHECK(cudaMalloc(&out, grid.x * block.x * 4));
    CHECK(cudaMemset(in, 1, grid.x * block.x * 4));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    const char* names[2] = {"fmaf  (FP32 流水线)", "__sinf (SFU)       "};
    float ms[2] = {0.f, 0.f};
    for (int k = 0; k < 2; ++k) {
        if (k == 0) fma_kernel<<<grid, block>>>(in, out, reps);   // 预热
        else        sfu_kernel<<<grid, block>>>(in, out, reps);
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 5; ++r) {
            if (k == 0) fma_kernel<<<grid, block>>>(in, out, reps);
            else        sfu_kernel<<<grid, block>>>(in, out, reps);
        }
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));

        CHECK(cudaEventElapsedTime(&ms[k], t0, t1));
        ms[k] /= 5;
        printf("%s  %8.3f ms   %8.1f Gop/s\n", names[k], ms[k],
               total / (ms[k] / 1e3) / 1e9);
    }
    printf("\n吞吐比 x%.1f -- 对应硬件配比 128 FP32 lane : 16 SFU = 8:1\n",
           ms[1] / ms[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
