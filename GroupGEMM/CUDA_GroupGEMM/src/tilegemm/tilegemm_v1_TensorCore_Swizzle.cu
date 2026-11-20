/*
 * tilegemm_v1.cu
 * Tensor Core tile GEMM：CTA tile -> warp tile -> m16n8k16 MMA 指令 tile。
 * 阅读顺序：共享内存 swizzle -> MMA 寄存器接口 -> 搬运、计算与写回。
 * NOTE: FP16/BF16 输入输出，FP32 累加；面向 SM80+，当前项目编译为 RTX 4090 的 SM89。
 *       此文件由 cutsom_gemm.cu 包含，在同一翻译单元实例化设备模板。
 */

/* 
 * 核心改动先让一个 tile 使用 Tensor Core，并设计匹配的共享内存布局。
 * 当前核心让每个线程负责若干输出元素，沿 K 逐个累加。
 * Tensor Core 则让一个 warp 协作完成小矩阵乘加，再把这些小矩阵拼成整个 CTA 的输出 tile。
 * 核心实现是建立 CTA tile → warp tile → MMA 指令 tile 三层分块。
 * 针对当前 RTX 4090，可以采用 ldmatrix 加载寄存器片段、`mma.sync` 执行矩阵乘加，
 * 例如 16×8×16 的指令形状。FP16/BF16 输入配合 FP32 累加完全可以使用 Tensor Core。 
 * 这部分可以放进新的 tile_gemm_v1()，首先复用现有专家调度。
 * W 的物理布局 [N,K] 本来就对应数学上列主序的 B [K,N]，不需要额外做整个权重矩阵的转置。
 * 此外，当前 w_tile[tile_k][tile_n] 的写入方式值得优先处理：在选中的 BLOCK_N=64、BLOCK_K=32 配置下，
 * 一个 warp 的相邻线程写地址相差 128 字节，按共享内存 bank 映射会落入同一个 bank。
 * v0 源码因此显示出明显的冲突风险。新的布局应同时满足连续搬运和 MMA 读取，例如采用合适的布局置换（swizzle）；
*/
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace group_gemm {
namespace {



// ===========================
// shared memory 布局与 MMA 指令
// ===========================

/*
 * 将逻辑 [row,k] 映射到共享内存的一维元素下标；X 的 row 是 m，W 的 row 是 n。
 * NOTE: 全局内存仍是 X[M,K] / W[N,K]，只有共享内存使用此置换。
 *       每 8 个 16-bit 元素构成一个 16-byte 段，ldmatrix 每次读取这样的一行。
 *       XOR 仅改变下标的 bit 3..5，因此段内的 8 个元素仍连续且保持顺序。
 *
 * PERF: bank = (元素下标 / 2) % 32；一个 16-byte 段占连续 4 个 bank。
 *       BK=32、k=0 时，逻辑 row=0..7 的段起始 bank 依次是：
 *         未置换： 0,16, 0,16, 0,16, 0,16
 *         置换后： 0,20, 8,28,16, 4,24,12
 *       后者覆盖 8 个互不重叠的 bank 组，匹配 ldmatrix 的 8 行读取。
 *       BK=16 时同样覆盖全部 8 组；其他 k 段只是再次置换这些组。
 *
 * WARNING: XOR 作用于完整线性下标，不只是 k；BK=16/32 时可能改变物理行。
 *          此映射在每 8 行内是一一对应的，不需要 padding；写入和读取必须共用它。
 *          v3 的 BK=64 正好每行 8 段，此时 XOR 只改变行内段号：固定 k 时，
 *          相邻 8 行仍分散在不同 bank 组，沿一行复制的 8 个段也互不重叠。
 */
template<int BLOCK_K>
__device__ __forceinline__ int tile_smem_offset_v1(int row, int k) {
    static_assert(BLOCK_K == 16 || BLOCK_K == 32 || BLOCK_K == 64, "unsupported shared row stride");
    return (row * BLOCK_K + k) ^ ((row & 7) << 3);
}


/*
 * 一个 warp 协作执行 D[16,8] = A[16,16] * B[16,8] + C[16,8]。
 * a/b/c: 当前线程持有的寄存器片段，并非完整小矩阵，也不是共享内存地址。
 * NOTE: 每线程 A 有 4 个 32-bit 寄存器（8 个输入），B 有 2 个（4 个输入），
 *       C/D 有 4 个 float；32 个线程合起来才覆盖完整的矩阵。
 *       "r" 传递打包的原始位，不能将 FP16/BF16 数值转换成整数后传入。
 *       "+f" 表示 FP32 寄存器既是输入 C 又是输出 D，沿 K 保留累加值。
 * WARNING: warp 的全部 32 个线程必须执行同一条 mma.sync，不能按有效 M/N 分支。
 */
template<typename T>
__device__ __forceinline__ void mma_m16n8k16_v1(
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    float (&c)[4]
) {
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>,
                  "v1 only supports FP16/BF16 inputs");

    // 类型分派在编译期完成；两条指令都使用 FP32 累加。
    if constexpr (std::is_same_v<T, __half>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
        );
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
        );
    }
}



// ===========================
// 两种布局共用的 Tensor Core tile GEMM
// ===========================

/*
 * 一个 CTA 计算当前专家 Y 的 [BLOCK_M,BLOCK_N] tile，专家调度沿用 v0。
 * x_group, w_group, y_group: [Device] 当前专家矩阵起点，行步长分别为 K/K/N。
 * M 是有效行数；offs_m/offs_n 是 tile 左上角的元素坐标，不是 tile 编号。
 * NOTE: W[N,K] 的地址 n*K+k 正好也是数学上列主序 B[K,N] 的地址，
 *       因此使用 MMA 的 .row.col，无需转置整个 W，也不需要转置共享内存中的 W。
 * WARNING: blockDim.x 必须等于 NUM_THREADS，整个 CTA 必须一起调用；
 *          尾块通过补零参与完整 MMA，仅在全局内存读写处判断有效范围。
 */
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__device__ __forceinline__ void tile_gemm_v1(
    const T* x_group,
    const T* w_group,
    T* y_group,
    int M,
    int N,
    int K,
    int offs_m,
    int offs_n
) {
    /*
     * 关于参数: x_group, w_group, y_group 为此次需要进行 GEMM 计算的数据 x, w, y起始地址。
     * offs_m, offs_n 是此次 CTA 负责的 Tile GEMM 对于此次需要计算的数据 x 在 m 维度和 w 在 n 维度的偏移 
     */
    // 三层分块：先把 CTA 输出分给各 warp，再把 warp 输出分成 16x8 指令 tile。
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    static_assert(NUM_THREADS >= 32 && NUM_THREADS <= 1024 && NUM_THREADS % 32 == 0,
                  "v1 requires complete warps");
    static_assert(BLOCK_M >= MMA_M && BLOCK_M % MMA_M == 0,
                  "CTA M must be a multiple of MMA M");
    static_assert(BLOCK_N >= MMA_N && BLOCK_N % MMA_N == 0,
                  "CTA N must be a multiple of MMA N");

    // NOTE: 优先沿 M 放置 warp，剩余 warp 沿 N 排列；兼容现有 128/256/512 线程配置。
    //       案例默认 BLOCK_M = 32, BLOCK_N = 64, BLOCK_K = 32, NUM_THREAD = 128
    //       那么可知 NUM_WARPS = 4, 如下可以计算得到 WARPS_M = 2, WARPS_N = 2.
    // 一个 warp 负责计算完一次 MMA
    // WARPS_M 表示 Tile 在 M 维度需要多个 MMA 完成
    // WARPS_N 表示 Tile 在 N 维度上需要多少个 MMA 完成 
    //       规定 WARPS_M * WARPS_N = NUM_WARPS
    //       在此处每个 warp 算 16x32, 一个 warp 的 16x32 由 N 方向的 4 个 16x8 MMA tile 拼成。
    constexpr int WARPS_M = BLOCK_M / MMA_M < NUM_WARPS ? BLOCK_M / MMA_M : NUM_WARPS;
    constexpr int WARPS_N = NUM_WARPS / WARPS_M;
    static_assert(NUM_WARPS % WARPS_M == 0 && BLOCK_M % WARPS_M == 0,
                  "warps must evenly partition CTA M");
    static_assert(BLOCK_N % WARPS_N == 0, "warps must evenly partition CTA N");

    // WARP_M 表示每个 Warp 在 Tile M 维度上需要负责多少行，在此处为 16
    // WARP_N 表示每个 Warp 在 Tile N 维度上需要负责多少列，在此处为 32
    constexpr int WARP_M = BLOCK_M / WARPS_M;
    constexpr int WARP_N = BLOCK_N / WARPS_N;
    static_assert(WARP_M % MMA_M == 0 && WARP_N % MMA_N == 0,
                  "warp tiles must contain complete MMA tiles");
    
    // MMA_TILES_M 表示在 MMA 层次上, 处理一个 Warp 负责的 warp_tile, 在 M 维度上需要多个 mma_tile 计算
    // 沿用上述案例，BM=32, WARPS_M=2, WARP_M=16, MMA_M=16 => MMA_TILES_M = 1
    constexpr int MMA_TILES_M = WARP_M / MMA_M;
    // MMA_TILES_N 表示在 MMA 层次上, 处理一个 Warp 负责的 warp_tile, 在 N 维度上需要多个 mma_tile 计算
    // 沿用上述案例，BM=64, WARPS_N=2, WARP_N=32, MMA_N=8 => MMA_TILES_M = 4
    constexpr int MMA_TILES_N = WARP_N / MMA_N;

    // 当前 warp 的输出矩形互不重叠；lane 决定片段中的元素，而非独立计算一个点积。
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;   // 看 thread 在哪个 warp
    const int lane_id = tid % 32;   // 看 thread 在 warp 内的 thread idx
    
    // warp_m 表示在 Tile 上 M 维度 MMA 级别的偏移
    const int warp_m = (warp_id / WARPS_N) * WARP_M;
    // warp_n 表示在 Tile 上 N 维度 MMA 级别的偏移
    const int warp_n = (warp_id % WARPS_N) * WARP_N;

    const T* x_tile_base = x_group + int64_t(offs_m) * K;
    const T* w_tile_base = w_group + int64_t(offs_n) * K;
    T* y_tile_base = y_group + int64_t(offs_m) * N + offs_n;
    // swizzle 使用一维存储；显式对齐保证每个 8 元素段满足 ldmatrix 的 16-byte 对齐。
    __shared__ __align__(16) T x_tile[BLOCK_M * BLOCK_K];
    __shared__ __align__(16) T w_tile[BLOCK_N * BLOCK_K];
    
    // 因为一个 warp 中的 32 个 thread 都会一起处理 mma 的计算，mma 最终要算出 16x8 的结果出来
    // 即每个 thread 负责 16x8 / 32 = 4 个元素。
    // 有 MMA_TILES_M * MMA_TILES_N 个 mma 结果需要被处理.
    float accumulator[MMA_TILES_M][MMA_TILES_N][4] = {};

    // 每轮只搬运一个 K 分块；v1 先建立同步的数据流，尚未重叠搬运与计算。
    for (int64_t offs_k = 0; offs_k < K; offs_k += BLOCK_K) {

        // CTA 中全部线程一起搬运 Tile x 数据
        for (int ele_idx = tid; ele_idx < BLOCK_M * BLOCK_K; ele_idx += NUM_THREADS) {
            const int tile_m = ele_idx / BLOCK_K;
            const int tile_k = ele_idx % BLOCK_K;
            const bool valid = int64_t(offs_m) + tile_m < M && offs_k + tile_k < K;
            // swizzle 只重排共享内存中的 16-byte 段。
            x_tile[tile_smem_offset_v1<BLOCK_K>(tile_m, tile_k)] = valid
                ? x_tile_base[int64_t(tile_m) * K + offs_k + tile_k]
                : static_cast<T>(0.0f);
        }

        // CTA 中全部线程一起搬运 Tile w 数据
        // PERF: W 同样沿物理 K 连续搬运，不再使用 v0 的 w_tile[k][n] 跨行写入。
        //       BK=32、BN=64 时，v0 相邻线程的 shared 地址相差 128 bytes；
        //       这里保留 [N,K] 的逻辑顺序，再由同一 swizzle 分散 ldmatrix 的行地址。
        for (int ele_idx = tid; ele_idx < BLOCK_N * BLOCK_K; ele_idx += NUM_THREADS) {
            const int tile_n = ele_idx / BLOCK_K;
            const int tile_k = ele_idx % BLOCK_K;
            const bool valid = int64_t(offs_n) + tile_n < N && offs_k + tile_k < K;
            w_tile[tile_smem_offset_v1<BLOCK_K>(tile_n, tile_k)] = valid
                ? w_tile_base[int64_t(tile_n) * K + offs_k + tile_k]
                : static_cast<T>(0.0f);
        }

        // WARNING: MMA 的 .sync 只协调 warp，不能代替 CTA 内生产者与消费者之间的 barrier。
        // 所以要等待 CTA 中线程搬运数据全部完成。
        __syncthreads();

        // BK = 32 需要沿 K 执行两轮 m16n8k16；BK = 16 只需要一轮。
        #pragma unroll
        for (int tile_k = 0; tile_k < BLOCK_K; tile_k += MMA_K) {
            // 对于一个 16 × 16 的 FP16/BF16 矩阵 A，总共有  256$ 个 16-bit 元素。
            // 整个 warp 有 32 个线程，平均每个线程负责 $256 / 32 = 8$ 个 16-bit 元素；
            // 而 1 个 uint32_t 正好打包 2 个 16-bit 元素，所以每个线程需要 4 个 uint32_t 寄存器（共 8 个元素）。
            uint32_t a_fragment[MMA_TILES_M][4];
            uint32_t b_fragment[MMA_TILES_N][2];

            // // ldmatrix 把共享内存里的 A 搬进 warp 各线程的寄存器，排成 mma.sync 要求的格式。
            // A[16,16] 拆成 4 个 8x8；x4 的四组 lane 各提供 8 个行首地址：
            //   lane  0..7  -> A[0..7,   0..7]，lane  8..15 -> A[8..15,  0..7]
            //   lane 16..23 -> A[0..7,  8..15]，lane 24..31 -> A[8..15, 8..15]
            // NOTE: 提供行地址的 lane 与最终接收元素的 lane 并不是同一映射。
            //       ldmatrix 把每行分给 4 个线程，每线程得到两个 16-bit 元素。
            #pragma unroll
            for (int mma_m = 0; mma_m < MMA_TILES_M; ++mma_m) {
                const int row = warp_m + mma_m * MMA_M + lane_id % 16;
                const int k = tile_k + (lane_id / 16) * 8;
                const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
                    x_tile + tile_smem_offset_v1<BLOCK_K>(row, k)));

                // shared PTX 使用共享内存地址空间
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_fragment[mma_m][0]), "=r"(a_fragment[mma_m][1]),
                      "=r"(a_fragment[mma_m][2]), "=r"(a_fragment[mma_m][3])
                    : "r"(address)
                    : "memory"
                );
            }

            // B[K,N] 已以 W[N,K] 形式存储：每个 W 行就是 B 的一个列。
            // x2 分别加载 W 的 K=0..7 和 K=8..15；这里刻意不使用 .trans。
            // NOTE: 每线程得到 B[2*(lane%4)+{0,1}, lane/4] 及 K+8 的两个值，
            //       正好匹配 .row.col MMA 的 B 片段；加 .trans 反而会改变所需顺序。
            #pragma unroll
            for (int mma_n = 0; mma_n < MMA_TILES_N; ++mma_n) {
                const int row = warp_n + mma_n * MMA_N + lane_id % 8;
                const int k = tile_k + ((lane_id / 8) % 2) * 8;
                const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
                    w_tile + tile_smem_offset_v1<BLOCK_K>(row, k)));

                // x2 只采用 lane 0..15 的地址；其余 lane 重复有效地址，仍一起执行指令。
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_fragment[mma_n][0]), "=r"(b_fragment[mma_n][1])
                    : "r"(address)
                    : "memory"
                );
            }

            // 同一个 A 片段复用于多个 N tile，同一个 B 片段复用于多个 M tile。
            #pragma unroll
            for (int mma_m = 0; mma_m < MMA_TILES_M; ++mma_m) {
                #pragma unroll
                for (int mma_n = 0; mma_n < MMA_TILES_N; ++mma_n) {
                    mma_m16n8k16_v1<T>(
                        a_fragment[mma_m], b_fragment[mma_n], accumulator[mma_m][mma_n]
                    );
                }
            }
        }

        // WARNING: 等所有 warp 读完当前共享内存，才能覆写下一轮 K 或下一个专家的 tile。
        __syncthreads();
    }

    // MMA 输出片段的固定映射：令 g=lane/4, t=lane%4，则四个 float 对应
    //   c[0] -> (g, 2t),     c[1] -> (g, 2t+1),
    //   c[2] -> (g+8, 2t),   c[3] -> (g+8, 2t+1)。
    // NOTE: 32 lanes * 4 floats = 16*8，完整覆盖一个 MMA tile，线程之间不会重复写。
    //       先加 warp 和 MMA tile 的偏移，再判断实际 M/N；只在最后转换回 FP16/BF16。
    #pragma unroll
    for (int mma_m = 0; mma_m < MMA_TILES_M; ++mma_m) {
        #pragma unroll
        for (int mma_n = 0; mma_n < MMA_TILES_N; ++mma_n) {
            #pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                const int tile_m = warp_m + mma_m * MMA_M + lane_id / 4 + (slot / 2) * 8;
                const int tile_n = warp_n + mma_n * MMA_N + (lane_id % 4) * 2 + slot % 2;
                if (int64_t(offs_m) + tile_m < M && int64_t(offs_n) + tile_n < N) {
                    y_tile_base[int64_t(tile_m) * N + tile_n] =
                        static_cast<T>(accumulator[mma_m][mma_n][slot]);
                }
            }
        }
    }
}

}  // namespace
}  // namespace group_gemm
