// 实验 3: Shared Memory Bank Conflict
// 对应正文: Shared Memory 的 32 个 bank, 每个 bank 宽 32bit (体育场大门)
//
//   nvcc -O3 -arch=native 03_bank_conflict.cu -o 03 && ./03
//
// bank 编号 = 地址 (float 下标) % 32。
//   stride = 1 : 一个 warp 的 32 个线程正好落在 32 个不同 bank, 无冲突
//   stride = 32: 32 个线程全部落在同一个 bank -> 32 路冲突, LSU 串行重放
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

// 4 个累加器 => 每 warp 同时有 4 个 load 在飞,
// 保证瓶颈落在 bank 吞吐而不是单个 warp 的访问延迟上
__global__ void smem_stride(float* __restrict__ out, int stride, int reps) {
    __shared__ float s[4096];
    int tid = threadIdx.x;
    for (int i = tid; i < 4096; i += blockDim.x) s[i] = i * 1.0f;
    __syncthreads();

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    for (int r = 0; r < reps; r += 4) {
        acc0 += s[(tid * stride + r)     & 4095];
        acc1 += s[(tid * stride + r + 1) & 4095];
        acc2 += s[(tid * stride + r + 2) & 4095];
        acc3 += s[(tid * stride + r + 3) & 4095];
    }
    out[tid] = acc0 + acc1 + acc2 + acc3;
}

int main() {
    int sm_count = 0;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    const int reps = 4000;
    const dim3 grid(sm_count * 4), block(1024);
    float* out;
    CHECK(cudaMalloc(&out, grid.x * block.x * 4));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    printf("stride   冲突路数   耗时\n");
    const int strides[6] = {1, 2, 4, 8, 16, 32};
    float base = 0.f;
    for (int k = 0; k < 6; ++k) {
        int stride = strides[k];
        smem_stride<<<grid, block>>>(out, stride, reps);   // 预热
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 5; ++r)
            smem_stride<<<grid, block>>>(out, stride, reps);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));

        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= 5;
        if (k == 0) base = ms;
        printf("%6d   %8d   %8.3f ms   (x%.1f)\n", stride, stride, ms,
               ms / base);
    }

    cudaFree(out);
    return 0;
}
