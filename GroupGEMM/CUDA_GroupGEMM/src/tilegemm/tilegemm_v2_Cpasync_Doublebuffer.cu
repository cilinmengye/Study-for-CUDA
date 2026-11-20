/*
 * tilegemm_v2.cu
 * 在 v1 的 Tensor Core tile 上增加 16-byte cp.async 搬运与双缓冲 K 流水线。
 * 阅读顺序：向量搬运与补零 -> 双缓冲预热 -> 预取/计算重叠 -> 等待与缓冲区复用。
 * NOTE: v1 保留为同步版本；这里复用它的 swizzle/MMA 指令封装，计算片段和写回映射不变。
 *       FP16/BF16 输入输出，FP32 累加，要求 SM80+；由 cutsom_gemm.cu 包含。
 *       cp.async 的完成和跨线程可见性是两件事，不能用 __syncthreads() 代替 wait_group。
 *       https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
 */
#pragma once

#include "tilegemm_v1_TensorCore_Swizzle.cu"

namespace group_gemm {
namespace {



// ===========================
// 16-byte 异步搬运与边界处理
// ===========================

/*
 * 将一个逻辑 [TILE_ROWS,BLOCK_K] 分块搬入调用者指定的共享内存 stage。
 * matrix_group: [Device] 当前专家 X[M,K] 或 W[N,K] 的全局内存起点。
 * tile: [Device, shared] 一个 stage 的起点，必须 16-byte 对齐，使用 v1 swizzle。
 * rows 是有效 M 或 N；offs_row/offs_k 是本分块在专家矩阵中的元素坐标。
 * WARNING: 此函数只发起复制，不 commit/wait；调用者必须在消费前等待整个 X/W 复制组。
 */

 // 基本核心改动都在此处，此处是 global ───── cp.async ─────→ shared 体现 cp.async 的具体实现
template<typename T, int TILE_ROWS, int BLOCK_K, int NUM_THREADS>
__device__ __forceinline__ void copy_tile_async_v2(
    const T* matrix_group,
    T* tile,
    int rows,
    int K,
    int offs_row,
    int64_t offs_k
) {
    constexpr int VECTOR_ELEMENTS = 8;
    static_assert(sizeof(T) * VECTOR_ELEMENTS == 16, "v2 copies eight 16-bit elements");
    static_assert(BLOCK_K % VECTOR_ELEMENTS == 0, "K tiles must contain whole vectors");

    // NOTE: cudaMalloc 的起点对齐不代表传入的子视图也对齐，所以实际检查专家起点。
    //       K%8==0 保证行步长是 16-byte 倍数，offs_k 和 tile_k 也都是 8 的倍数。
    //       因而此分支中所有有效源地址均满足 16-byte 对齐，而不只是第一个地址。
    const bool aligned = (reinterpret_cast<uintptr_t>(matrix_group) & 15) == 0 && K % 8 == 0;
    if (aligned) {
        // PERF: v1 每线程每次搬一个元素；v2 每次直接搬 8 个，不经通用寄存器中转。
        //       相邻线程负责相邻的 K 向量；向量不足线程数时允许部分线程不发起复制。
        #pragma unroll
        for (int vector_idx = threadIdx.x;
             vector_idx < TILE_ROWS * BLOCK_K / VECTOR_ELEMENTS;
             vector_idx += NUM_THREADS) {
            const int row = vector_idx / (BLOCK_K / VECTOR_ELEMENTS);
            const int tile_k = (vector_idx % (BLOCK_K / VECTOR_ELEMENTS)) * VECTOR_ELEMENTS;
            const bool valid = int64_t(offs_row) + row < rows && offs_k + tile_k < K;

            // NOTE: 接口 K 为 8 的倍数，尾块可以不足 BK，但每个 8 元素段必然全有效或全无效。
            //       例如 BK=32,K=40，最后一块的四段 src_bytes 分别为 16,0,0,0。
            //       src_bytes=0 仍写满 16 字节的零，不能只跳过复制而留下旧 stage 数据。
            // WARNING: 无效 M/N/K 段使用有效占位地址且完全不读源，避免 NaN padding 和越界。
            const T* source = valid
                ? matrix_group + (int64_t(offs_row) + row) * K + offs_k + tile_k
                : matrix_group;
            const int src_bytes = valid ? 16 : 0;

            // NOTE: v1 swizzle 的 XOR 只改变 bit 3..5，段内 8 个元素的连续性不变。
            //       shared 数组起点对齐，置换后的元素下标也是 8 的倍数，目标因此对齐。
            const uint32_t destination = static_cast<uint32_t>(__cvta_generic_to_shared(
                tile + tile_smem_offset_v1<BLOCK_K>(row, tile_k)));
            
            // cp.async 发起复制
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                :: "r"(destination), "l"(source), "r"(src_bytes)
                : "memory"
            );
        }
    } else {
        // NOTE: 未对齐子视图使用标量安全路径，不向 cp.async 提交未对齐地址。
        //       这条路径不提供异步重叠；保持原有接口的正确性，正常对齐输入才走快路径。
        //       逐元素边界检查也避免在非 8 倍数的行步长下跨行读取。
        for (int element = threadIdx.x; element < TILE_ROWS * BLOCK_K; element += NUM_THREADS) {
            const int row = element / BLOCK_K;
            const int tile_k = element % BLOCK_K;
            const bool valid = int64_t(offs_row) + row < rows && offs_k + tile_k < K;
            tile[tile_smem_offset_v1<BLOCK_K>(row, tile_k)] = valid
                ? matrix_group[(int64_t(offs_row) + row) * K + offs_k + tile_k]
                : static_cast<T>(0.0f);
        }
    }
}



// ===========================
// Tensor Core tile 与双缓冲 K 流水线
// ===========================

/*
 * 一个 CTA 计算当前专家的 [BLOCK_M,BLOCK_N] 输出 tile，外层专家调度沿用 v1。
 * x_group, w_group, y_group: [Device] 当前专家矩阵起点，行步长分别为 K/K/N。
 * M 是有效行数，offs_m/offs_n 是输出 tile 左上角的元素坐标。
 * WARNING: 整个 CTA 一起调用，blockDim.x 必须为 NUM_THREADS，不能按 M/N 有效性提前退出。
 * NOTE: 两个 stage 交替经历「可写 -> 复制中 -> 可读 -> 所有 warp 读完 -> 可写」。
 *       多一个数组只提供存储空间；先发起下一块、再算当前块、最后才等待，才形成重叠。
 */
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__device__ __forceinline__ void tile_gemm_v2(
    const T* x_group,
    const T* w_group,
    T* y_group,
    int M,
    int N,
    int K,
    int offs_m,
    int offs_n
) {
    // 三层分块与 v1 相同：CTA -> warp -> m16n8k16，避免把流水线与计算布局改动混在一起。
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    static_assert(NUM_THREADS >= 32 && NUM_THREADS <= 1024 && NUM_THREADS % 32 == 0,
                  "v2 requires complete warps");
    static_assert(BLOCK_M >= MMA_M && BLOCK_M % MMA_M == 0,
                  "CTA M must be a multiple of MMA M");
    static_assert(BLOCK_N >= MMA_N && BLOCK_N % MMA_N == 0,
                  "CTA N must be a multiple of MMA N");

    constexpr int WARPS_M = BLOCK_M / MMA_M < NUM_WARPS ? BLOCK_M / MMA_M : NUM_WARPS;
    constexpr int WARPS_N = NUM_WARPS / WARPS_M;
    static_assert(NUM_WARPS % WARPS_M == 0 && BLOCK_M % WARPS_M == 0,
                  "warps must evenly partition CTA M");
    static_assert(BLOCK_N % WARPS_N == 0, "warps must evenly partition CTA N");
    constexpr int WARP_M = BLOCK_M / WARPS_M;
    constexpr int WARP_N = BLOCK_N / WARPS_N;
    static_assert(WARP_M % MMA_M == 0 && WARP_N % MMA_N == 0,
                  "warp tiles must contain complete MMA tiles");
    constexpr int MMA_TILES_M = WARP_M / MMA_M;
    constexpr int MMA_TILES_N = WARP_N / MMA_N;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = (warp_id / WARPS_N) * WARP_M;
    const int warp_n = (warp_id % WARPS_N) * WARP_N;
    T* y_tile_base = y_group + int64_t(offs_m) * N + offs_n;

    // NOTE: 核心改动 1：v1 的一套 X/W 变为两套，读当前 stage 时不会被预取覆盖。
    //       每个 stage 的跨度都是 16-byte 倍数，切换 stage 不破坏 cp.async/ldmatrix 对齐。
    //       BM=32,BN=64,BK=32 时 shared 从 6 KiB 增至 12 KiB；驻留 CTA 数由现有 occupancy 查询决定。
    constexpr int STAGES = 2;
    __shared__ __align__(16) T x_tiles[STAGES][BLOCK_M * BLOCK_K];
    __shared__ __align__(16) T w_tiles[STAGES][BLOCK_N * BLOCK_K];
    float accumulator[MMA_TILES_M][MMA_TILES_N][4] = {};

    // NOTE: 核心改动 2（预热）：先准备 K=0 的 stage 0，此时还没有可重叠的计算。
    //       X 与 W 的复制放入同一个 per-thread group；即使线程没有搬运任务也参与 commit。
    copy_tile_async_v2<T, BLOCK_M, BLOCK_K, NUM_THREADS>(x_group, x_tiles[0], M, K, offs_m, 0);
    copy_tile_async_v2<T, BLOCK_N, BLOCK_K, NUM_THREADS>(w_group, w_tiles[0], N, K, offs_n, 0);
    
    // 当 cp.async 发起复制后，cp.async.commit_group 的意思是：
    // 把当前线程此前已发起、尚未提交的所有 cp.async 复制归为一组，供后续等待。
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    // 我发起的复制完成了吗？ 每个线程等自己提交的异步复制完成，也就是自己负责搬的那部分 X/W 已经到了下一块缓冲区。
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");

    // WARNING: wait_group 只等待本线程提交的复制。全体线程 wait 后再做 CTA barrier，
    //          才能让消费者 warp 安全读取其他线程搬来的 X/W；标量安全路径也依赖此 barrier。
    // 此处是等待整个 CTA 中的每个线程都是在读完当前块、等完自己的复制后才到这里。
    __syncthreads();

    // 经过上述同步点后，stage = 0 即缓存区 0 以及加载好 Tile x, Tile w 数据。

    int read_stage = 0;
    for (int64_t offs_k = 0; offs_k < K; offs_k += BLOCK_K) {
        // NOTE: 核心改动 3（稳态）：先发起 k+1，再计算 k，中间刻意不等待 k+1。
        //       时序：copy(0)->wait | copy(1) + MMA(0)->wait | copy(2) + MMA(1)->wait | ...
        //       + 表示复制可以在后台推进；并不保证所有数据搬运延迟都能被当前块计算覆盖。
        const int write_stage = read_stage ^ 1;
        const int64_t next_k = offs_k + BLOCK_K;
        
        // 先发起对下一个阶段需要使用到的 Tile x, Tile w 加载。
        if (next_k < K) {
            // WARNING: 此 stage 上次的所有 ldmatrix 已在上一轮末尾 barrier 前完成。
            //          第一次使用 stage 1 时它尚未被读取，因此同样可以安全写入。
            copy_tile_async_v2<T, BLOCK_M, BLOCK_K, NUM_THREADS>(
                x_group, x_tiles[write_stage], M, K, offs_m, next_k
            );
            copy_tile_async_v2<T, BLOCK_N, BLOCK_K, NUM_THREADS>(
                w_group, w_tiles[write_stage], N, K, offs_n, next_k
            );
            // 提交等待线程组
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        // v1 的计算主体保持相同；唯一变化是 ldmatrix 的基址选择当前 read_stage。
        const T* x_tile = x_tiles[read_stage];
        const T* w_tile = w_tiles[read_stage];
        #pragma unroll
        for (int tile_k = 0; tile_k < BLOCK_K; tile_k += MMA_K) {
            uint32_t a_fragment[MMA_TILES_M][4];
            uint32_t b_fragment[MMA_TILES_N][2];

            // A[16,16] 的四个 8x8 子块：lane/16 选择 K 半块，lane%16 选择行。
            #pragma unroll
            for (int mma_m = 0; mma_m < MMA_TILES_M; ++mma_m) {
                const int row = warp_m + mma_m * MMA_M + lane_id % 16;
                const int k = tile_k + (lane_id / 16) * 8;
                const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
                    x_tile + tile_smem_offset_v1<BLOCK_K>(row, k)));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_fragment[mma_m][0]), "=r"(a_fragment[mma_m][1]),
                      "=r"(a_fragment[mma_m][2]), "=r"(a_fragment[mma_m][3])
                    : "r"(address)
                    : "memory"
                );
            }

            // NOTE: W[N,K] 已对应列主序 B[K,N]；与 v1 相同，x2 不加 .trans。
            #pragma unroll
            for (int mma_n = 0; mma_n < MMA_TILES_N; ++mma_n) {
                const int row = warp_n + mma_n * MMA_N + lane_id % 8;
                const int k = tile_k + ((lane_id / 8) % 2) * 8;
                const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
                    w_tile + tile_smem_offset_v1<BLOCK_K>(row, k)));
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b_fragment[mma_n][0]), "=r"(b_fragment[mma_n][1])
                    : "r"(address)
                    : "memory"
                );
            }

            // 同一 A/B 片段复用于多个输出指令 tile，累加器跨 K 迭代保留。
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

        // NOTE: 核心改动 4（交接/排空）：最多一个未完成组，所以双缓冲用 wait_group 0。
        //       wait_group 1 会允许唯一的下一块仍在复制，随后读它就会发生竞态。
        //       等待放在计算之后，只有尚未被计算覆盖的复制时间会在这里形成停顿。
        // 即可以理解为 前一缓存区中数据计算完成，即将用到此次异步加载的下一缓冲区，先要确保下一缓冲区加载完成
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");

        // WARNING: 此 barrier 同时保证两件事：
        //          1. 下一块的所有生产者都已 wait，交换后可以跨 warp 读取新 stage。
        //          2. 当前块的所有消费者都已读完，下一轮可以覆盖旧 stage。
        //          最后一轮不再预取，但仍保留此 barrier，保护 persistent 调度的下一个输出 tile。
        __syncthreads();
        read_stage = write_stage;
    }

    // 写回与 v1 相同：每个 lane 的四个 float 对应 (g,2t)、(g,2t+1)、(g+8,2t)、(g+8,2t+1)。
    // NOTE: 无效行只在 shared 中补零参与 MMA，不会写回全局 masked padding。
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
