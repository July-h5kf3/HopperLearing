# Assignment 01

对应 lecture 1（GPU 体系结构）。四个小实验分别验证正文中的四个硬件概念，每个实验是一个自包含的 `.cu` 文件。

## 环境

- 任意一张 NVIDIA GPU（实验用通用特性，不依赖 Hopper 特有功能；趋势以 H100 为准）
- CUDA Toolkit（`nvcc`）

编译运行：

```bash
nvcc -O3 -arch=native 01_coalescing.cu -o 01 && ./01
```

（老版本 `nvcc` 若没有 `-arch=native`，换成本机算力，如 H100 用 `-arch=sm_90`。）

## 实验 1：Coalescing 决定有效带宽（`01_coalescing.cu`）

对应正文：HBM3 / L2（128B cache line、32B sector）/ LSU 的 coalescing。

复制同样大小的数据，一个 kernel 连续访存，另一个 stride=32 访存：

```cuda
// 连续: warp 内 32 个线程地址连续, 合并成最少的 sector
out[i] = in[i];

// strided: warp 内 32 个线程落在 32 个不同的 128B 缓存行上
out[i] = in[(long long)i * 32 % n];
```

预期：strided 版本有效带宽掉近一个数量级。sector 被取上来 32B 却只用 4B，就是 fig1.2 那张反面例子的实测版。

H100 实测：coalesced 2345 GB/s vs strided 341 GB/s（约 6.9 倍）。

## 实验 2：Occupancy 与 Latency Hiding（`02_occupancy.cu`）

对应正文：Warp / Warp Scheduler / 延迟隐藏。

固定总工作量的拷贝 kernel，`#pragma unroll 1` 强制每个 warp 同一时刻只允许一个 load 在飞（store 依赖 load 的结果）。此时在飞内存请求数 ≈ 常驻 warp 数。程序扫 1~32 blocks/SM：

预期：有效带宽先随 blocks/SM 近似线性增长，到某个拐点后变平——拐点即延迟被完全隐藏的位置，也和 latency_hiding.gif 演示的状态对应。

H100 实测：620 → 1128 → 1888 → 2418 GB/s（1/2/4/8 blocks/SM），8 blocks/SM（约 50% occupancy）后进入平台，拐点非常清晰。

## 实验 3：Shared Memory 的 Bank Conflict（`03_bank_conflict.cu`）

对应正文：32 bank × 32bit（体育场大门）。

bank 编号 = 地址（float 下标）% 32。用不同 stride 读 shared memory 做累加：

- stride=1：一个 warp 正好打满 32 个 bank，无冲突
- stride=32：32 个线程全部落在 bank 0，32 路冲突

预期：耗时随 stride 增大线性恶化，stride=32 时约为无冲突的 32 倍量级（LSU 对 32 路冲突的重放）。

H100 实测：x1.0 → x1.8 → x3.7 → x7.3 → x14.7 → x29.3，与冲突路数几乎完美线性。

## 实验 4：SFU 的吞吐瓶颈（`04_sfu.cu`）

对应正文：每 SM 只有 16 个 SFU vs 128 个 FP32 lane（8:1）。

两个 kernel 循环次数完全相同，唯一区别是循环体：

```cuda
c = fmaf(a, b, c);    // 走 FP32 流水线
c = __sinf(a + c);    // MUFU 指令, 走 SFU
```

预期：吞吐比接近 8:1。H100 实测约 6.3x——FP32 kernel 打不满 128 lane 的峰值，而 `__sinf` 循环体里的 FADD 白嫖了空闲的 FP32 流水线，所以比值略低于理想上限。方向不变：SFU 的吞吐确实只有 FP32 的几分之一。

> 注意必须用 `__sinf` intrinsic：普通的 `sinf()` 默认会被编译成软件多项式（一堆 FMA），那样测的还是 FP32 流水线，根本碰不到 SFU。
