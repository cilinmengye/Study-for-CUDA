/*
 * tilegemm_v0.cu
 * 原始 tile GEMM 计算核心；仅将函数名改为 tile_gemm_v0，函数体和注释原样迁移。
 * NOTE: 此文件保存设备函数模板，由 cutsom_gemm.cu 包含，不作为独立翻译单元编译。
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace group_gemm {
namespace {



// ===========================
// 两种布局共用的 tile GEMM
// ===========================

/*
 * 一个 CTA 协作计算当前专家 Y 上的一个 [BLOCK_M,BLOCK_N] tile。
 * x_group, w_group, y_group: [Device] 当前专家的矩阵起点，行步长分别为 K/K/N。
 * M: 当前专家的有效行数，不是 masked 的物理 capacity。
 * offs_m, offs_n: tile 左上角的元素坐标，不是 tile 编号。
 * WARNING: blockDim.x 必须等于 NUM_THREADS，且整个 block 一起调用此函数；
 *          内部的 __syncthreads() 不能置于只有部分线程进入的分支中。
 * NOTE: __device__ 函数复用当前 CTA 的线程与共享内存，不启动新的 kernel。
 */
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__device__ __forceinline__ void tile_gemm_v0(
    const T* x_group,
    const T* w_group,
    T* y_group,
    int M,
    int N,
    int K,
    int offs_m,
    int offs_n
) {
    // =============== 开始执行 tile GEMM ==================
    // 现在 Tile GEMM 层次在 Y (M, N) 上计算出 Tile (BLOCK_M, BLOCK_N)
    // 有 NUM_THREAD = NUM_WARP * 32 参与此次 Tile 计算
    // NOTE: 原注释中的 NUM_THREAD/NUM_WARP 对应这里的 NUM_THREADS/线程束数量。

    // N 和 K 是运行时值，不能写到模板实参中。
    // NOTE: 连续行存储的 STRIDE_XROW/STRIDE_WROW/STRIDE_YROW 分别为 K/K/N，列步长均为 1。
    // 建立当前 tile 的地址和存储；矩阵偏移使用 64-bit 元素索引。
    const int tid = threadIdx.x;

    // 得到 x,w,y 在 Tile 上的起始地址
    const T* x_tile_base = x_group + int64_t(offs_m) * K;
    const T* w_tile_base = w_group + int64_t(offs_n) * K;
    T* y_tile_base = y_group + int64_t(offs_m) * N + offs_n;
    __shared__ T x_tile[BLOCK_M][BLOCK_K];
    __shared__ T w_tile[BLOCK_K][BLOCK_N];

    // 此处采用 persistent tile GEMM 方案, 因为我们有 NUM_THREAD 个线程参与 tile GEMM 计算
    // tile 总共需要计算出的元素为 BLOCK_M * BLOCK_N, 那么每个线程可以负责
    // C_DIV(BLOCK_M * BLOCK_N, NUM_THREAD)个 tile 元素
    // NOTE: C_DIV 表示向上取整，下面的 elements_per_thread 使用同一个计算公式。
    constexpr int tile_elements = BLOCK_M * BLOCK_N;
    constexpr int elements_per_thread = (tile_elements + NUM_THREADS - 1) / NUM_THREADS;

    // 保持 float, 因为需要避免 bf16/fp16 进行低精度的矩阵乘法
    float accumulator[elements_per_thread] = {0.0f};

    // Tile GEMM 在 K 维度上的处理次数
    // 每轮沿 K 维加载一对 tile，然后累加到线程私有的 FP32 寄存器中。
    for (int64_t offs_k = 0; offs_k < K; offs_k += BLOCK_K) {
        // NUM_THREADS 所有线程协作读取 X 的一个分块 (BLOCK_M, BLOCK_K)
        // WARNING: 先检查有效 M，再读取 X；masked padding 中可能是 NaN。
        for (int ele_idx = tid; ele_idx < BLOCK_M * BLOCK_K; ele_idx += NUM_THREADS) {
            const int tile_m = ele_idx / BLOCK_K;
            const int tile_k = ele_idx % BLOCK_K;

            // load x 相关数据到 shared memory
            const bool valid = int64_t(offs_m) + tile_m < M && offs_k + tile_k < K;
            x_tile[tile_m][tile_k] = valid
                ? x_tile_base[int64_t(tile_m) * K + offs_k + tile_k]
                : static_cast<T>(0.0f);
        }

        // NUM_THREADS 所有线程协作读取 W 的一个分块 (BLOCK_K, BLOCK_N)
        // 需要注意的是这里 W shape (N, K) 而非 (K, N)
        // W 的物理行是 N，装入共享内存时按 [K,N] 索引供点积使用。
        for (int ele_idx = tid; ele_idx < BLOCK_N * BLOCK_K; ele_idx += NUM_THREADS) {
            // 得到在 (BLOCK_M, BLOCK_N) 这个 Tile 上的坐标
            // NOTE: 这里具体展开的是 W 分块的 (tile_n, tile_k)，对应共享内存 [BLOCK_K,BLOCK_N]。
            const int tile_n = ele_idx / BLOCK_K;
            const int tile_k = ele_idx % BLOCK_K;
            const bool valid = int64_t(offs_n) + tile_n < N && offs_k + tile_k < K;
            w_tile[tile_k][tile_n] = valid
                ? w_tile_base[int64_t(tile_n) * K + offs_k + tile_k]
                : static_cast<T>(0.0f);
        }

        // 等待全部将完整数据写入共享内存
        // WARNING: 所有协作加载完成后，其他线程才能读取共享内存。
        __syncthreads();

        // (BLOCK_M, BLOCK_N) 的 Tile GEMM 计算
        // 每个 thread 负责 y tiling 矩阵上若干个元素
        // 每个线程独占若干输出元素，操作数先转 FP32 再进行乘加。
        #pragma unroll // 展开固定次数的循环，方便编译器处理累加数组。
        for (int slot = 0; slot < elements_per_thread; ++slot) {
            const int ele_idx = tid + NUM_THREADS * slot;
            if (ele_idx < tile_elements) {
                // 仅计算分配给当前线程的有效 tile 元素。
                const int tile_m = ele_idx / BLOCK_N;
                const int tile_n = ele_idx % BLOCK_N;
                for (int ele_k = 0; ele_k < BLOCK_K; ++ele_k) {
                    // NOTE: 两个操作数先转换为 FP32，避免先进行低精度乘法。
                    accumulator[slot] +=
                        static_cast<float>(x_tile[tile_m][ele_k]) *
                        static_cast<float>(w_tile[ele_k][tile_n]);
                }
            }
        }

        // 所有线程完成后，才能加载下一个 K 分块。
        // WARNING: 所有线程读完本轮数据后才能覆写共享内存；也保护下一个输出 tile。
        __syncthreads();
    }

    // NUM_THREADS 所有线程协作将寄存器 accumulator 中的结果写入最终的 y 地址中。
    // 注意因为 accmulator 中保存的是若干个 (mi, nj) 元素结果，
    // 所以其经过上述累积和得到的是完整的结果。
    // 只写有效输出；masked 的剩余容量必须逐位保持原值。
    #pragma unroll
    for (int slot = 0; slot < elements_per_thread; ++slot) {
        const int ele_idx = tid + NUM_THREADS * slot;
        const int tile_m = ele_idx / BLOCK_N;
        const int tile_n = ele_idx % BLOCK_N;
        if (ele_idx < tile_elements &&
            int64_t(offs_m) + tile_m < M && int64_t(offs_n) + tile_n < N) {
            y_tile_base[int64_t(tile_m) * N + tile_n] = static_cast<T>(accumulator[slot]);
        }
    }
    // =============== 结束执行 tile GEMM ==================
}

}  // namespace
}  // namespace group_gemm
