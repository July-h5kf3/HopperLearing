# Assignment 01：把硬件数字跑出来

配合 Lecture 1（GPU 体系结构）。四个实验分别验证正文的四个硬件概念——Coalescing、Occupancy 与 Latency Hiding、Bank Conflict、SFU 吞吐。每个实验是一个自包含的 `.cu` 文件：

```bash
nvcc -O3 -arch=native 01_coalescing.cu -o 01 && ./01
```

实验只用到通用 CUDA 特性，不要求 H100：不同卡上绝对数字会不同，但趋势是一样的（下文预期现象以 H100 为参照）。

## 任务

### 实验 1：Coalescing 决定有效带宽（`01_coalescing.cu`）

复制同样 256MB 的数据，一个 kernel 让 warp 内 32 个线程访问连续地址，另一个让每个线程都跨过一个 128B cache line。

- 跑两个 kernel，记录各自的有效带宽。
- 思考：硬件实际搬的字节数差了多少倍？真正有用的字节差了多少倍？（对应 fig1.1 / fig1.2）
- 选做：用 `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 数出两个版本各取了多少 sector，验证你的推断。

预期：strided 版本有效带宽掉一个数量级左右（有用的字节只剩 1/32）。

### 实验 2：Occupancy 与 Latency Hiding（`02_occupancy.cu`）

总工作量固定，但 `#pragma unroll 1` 强制每个 warp 同一时刻只有一个 load 在飞，于是在飞请求数 ≈ 常驻 warp 数。程序会扫 1~32 blocks/SM 打印带宽曲线。

- 找到带宽开始变平的拐点，对应多少 blocks/SM？
- 用 Little's Law 粗算：拐点处 在飞请求数 × 每请求字节数 / 有效带宽 ≈ 内存延迟，是否落在几百 ns 量级？

预期：带宽先近似线性增长，饱和后变平——变平的拐点就是内存延迟被完全隐藏的位置。

### 实验 3：Shared Memory 的 Bank Conflict（`03_bank_conflict.cu`）

bank 编号 = 地址（float 下标）% 32，用 stride 控制冲突路数，读 shared memory 做累加。

- 记录 stride = 1 / 2 / 4 / 8 / 16 / 32 的耗时比。
- 思考：耗时比和冲突路数是什么关系？stride=32 时 LSU 实际发生了什么？

预期：耗时基本按冲突路数线性上升，stride=32（32 路冲突）约为 stride=1 的几十倍。

### 实验 4：SFU 的吞吐瓶颈（`04_sfu.cu`）

两个 kernel 循环次数完全相同，唯一区别是循环体一条走 FP32 流水线（`fmaf`）、一条走 SFU（`__sinf`）。

- 记录两者的吞吐比，和硬件配比 128 FP32 lane : 16 SFU 对一下。
- 思考：为什么必须用 `__sinf` 而不能用 `sinf()`？（提示：把 `sinf()` 换进去跑一次，再想想 `sinf()` 的默认实现是什么。）

预期：吞吐比约 8:1。

## 提交

不需要交代码，把四个实验在你机器上的输出贴出来，并回答各实验下的思考问题即可。
