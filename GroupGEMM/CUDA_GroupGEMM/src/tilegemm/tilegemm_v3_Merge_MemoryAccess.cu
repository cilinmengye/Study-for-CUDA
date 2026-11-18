/*
 * tilegemm_v3_Merge_MemoryAccess.cu
 * 分组 GEMM 的 CTA 核心。K 循环仍在 FP32 寄存器中累加；只在最后一次写回时舍入。
 * 阅读顺序：共享内存的两个生命周期、B 片段的读取、MMA 结果的重排与向量写回。
 */
#pragma once

#include "tilegemm_v2_Cpasync_Doublebuffer.cu"

namespace group_gemm {
namespace {

/*
 * x_group, w_group, y_group: [Device] 当前专家矩阵起点，行步长分别为 K/K/N。
 * 整个 CTA 计算从 (offs_m, offs_n) 开始的一个输出 tile。
 * WARNING: 所有 warp 必须经过相同的 CTA barrier，即使当前输出行无效。
 */
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__device__ __forceinline__ void tile_gemm_v3(
    const T* x_group,
    const T* w_group,
    T* y_group,
    int M,
    int N,
    int K,
    int offs_m,
    int offs_n
) {
    // BM/BN 决定一份输入供多少个输出使用。例如 FP16 的 128x128 CTA 相比
    // 64x64 CTA，同一 K 长度下计算量为 4 倍，而单 CTA 输入只需 2 倍。
    // 更大的 warp tile 会多占 FP32 累加寄存器，实际配置由各场景的 GPU 计时选择。
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    static_assert(NUM_THREADS >= 32 && NUM_THREADS <= 1024 && NUM_THREADS % 32 == 0,
                  "v3 requires complete warps");
    static_assert(BLOCK_M >= MMA_M && BLOCK_M % MMA_M == 0,
                  "CTA M must be a multiple of MMA M");
    static_assert(BLOCK_N >= MMA_N && BLOCK_N % MMA_N == 0,
                  "CTA N must be a multiple of MMA N");

    // Prefill 的 128×128、128 线程配置沿 M 放置 4 个 warp，每个 warp 算 32×128。
    // 在 RTX 4090 的 FP16/BF16 Prefill 实测中，这个布局的 kernel 延迟最低。
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

    // NOTE: K 循环期间按 [X0|X1|W0|W1] 使用 shared；最后一次 wait + CTA barrier
    //       之后输入块已读完，同一块内存可供 Y 的 epilogue 重排使用。取两阶段容量
    //       的较大值而非相加，避免仅为写回再降低驻留 CTA 数。
    constexpr int X_SIZE = BLOCK_M * BLOCK_K;
    constexpr int W_SIZE = BLOCK_N * BLOCK_K;
    constexpr int INPUT_SIZE = 2 * (X_SIZE + W_SIZE);
    constexpr int OUTPUT_SIZE = BLOCK_M * BLOCK_N;
    __shared__ __align__(16) T storage[INPUT_SIZE > OUTPUT_SIZE ? INPUT_SIZE : OUTPUT_SIZE];
    T* x_tiles = storage;
    T* w_tiles = storage + 2 * X_SIZE;
    float accumulator[MMA_TILES_M][MMA_TILES_N][4] = {};

    // NOTE: v2 双缓冲预热：先准备 K=0 的 stage 0，此时还没有可重叠的计算。
    //       X 与 W 的复制放入同一个 per-thread group；即使线程没有搬运任务也参与 commit。
    copy_tile_async_v2<T, BLOCK_M, BLOCK_K, NUM_THREADS>(x_group, x_tiles, M, K, offs_m, 0);
    copy_tile_async_v2<T, BLOCK_N, BLOCK_K, NUM_THREADS>(w_group, w_tiles, N, K, offs_n, 0);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");

    // WARNING: wait_group 只等待本线程提交的复制。全体线程 wait 后再做 CTA barrier，
    //          才能让消费者 warp 安全读取其他线程搬来的 X/W；标量安全路径也依赖此 barrier。
    __syncthreads();

    int read_stage = 0;
    for (int64_t offs_k = 0; offs_k < K; offs_k += BLOCK_K) {
        // NOTE: 双缓冲稳态：先发起 k+1，再计算 k，中间刻意不等待 k+1。
        //       时序：copy(0)->wait | copy(1) + MMA(0)->wait | copy(2) + MMA(1)->wait | ...
        //       + 表示复制可以在后台推进；并不保证所有数据搬运延迟都能被当前块计算覆盖。
        const int write_stage = read_stage ^ 1;
        const int64_t next_k = offs_k + BLOCK_K;
        if (next_k < K) {
            // WARNING: 此 stage 上次的所有 ldmatrix 已在上一轮末尾 barrier 前完成。
            //          第一次使用 stage 1 时它尚未被读取，因此同样可以安全写入。
            copy_tile_async_v2<T, BLOCK_M, BLOCK_K, NUM_THREADS>(
                x_group, x_tiles + write_stage * X_SIZE, M, K, offs_m, next_k
            );
            copy_tile_async_v2<T, BLOCK_N, BLOCK_K, NUM_THREADS>(
                w_group, w_tiles + write_stage * W_SIZE, N, K, offs_n, next_k
            );
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        // 本轮从 read_stage 读 A；相邻两片 B 的地址分组见下方 x4 加载。
        const T* x_tile = x_tiles + read_stage * X_SIZE;
        const T* w_tile = w_tiles + read_stage * W_SIZE;
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

            // v2 用一次 x2 读一个 8 列的 B 片段。v3 用 x4 的后两组 lane 地址
            // 同时读相邻的下一个片段：每 8 个 lane 提供一个 8x8 子矩阵的行首。
            //   lane 0..7 / 8..15   -> B0 的低/高 K 半块
            //   lane 16..23 / 24..31 -> B1 的低/高 K 半块
            // 因此前两个返回寄存器仍可原样喂给 B0 的 MMA，后两个喂给 B1。
            // PERF: 两片段读取的字节数相同，warp 发射的 ldmatrix 从两条降到一条。
            constexpr int B_STEP = MMA_TILES_N % 2 == 0 ? 2 : 1;
            #pragma unroll
            for (int n = 0; n < MMA_TILES_N; n += B_STEP) {
                const int row = warp_n + (n + (B_STEP == 2 ? lane_id / 16 : 0)) * MMA_N + lane_id % 8;
                const int k = tile_k + ((lane_id / 8) % 2) * 8;
                const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
                    w_tile + tile_smem_offset_v1<BLOCK_K>(row, k)));
                if constexpr (B_STEP == 2) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b_fragment[n][0]), "=r"(b_fragment[n][1]),
                                   "=r"(b_fragment[n + 1][0]), "=r"(b_fragment[n + 1][1])
                                 : "r"(address) : "memory");
                } else {
                    // 例如某些窄 warp tile 只有 8 列，此时没有第二个片段可以合并。
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                                 : "=r"(b_fragment[n][0]), "=r"(b_fragment[n][1])
                                 : "r"(address) : "memory");
                }
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

        // NOTE: 缓冲区交接与排空：最多一个未完成组，所以双缓冲用 wait_group 0。
        //       wait_group 1 会允许唯一的下一块仍在复制，随后读它就会发生竞态。
        //       等待放在计算之后，只有尚未被计算覆盖的复制时间会在这里形成停顿。
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");

        // WARNING: 此 barrier 同时保证两件事：
        //          1. 下一块的所有生产者都已 wait，交换后可以跨 warp 读取新 stage。
        //          2. 当前块的所有消费者都已读完，下一轮可以覆盖旧 stage。
        //          最后一轮不再预取，但仍保留此 barrier，保护 persistent 调度的下一个输出 tile。
        __syncthreads();
        read_stage = write_stage;
    }

    // v2 的一个 STG.U16 让 lane 0..3 分别写同一行的 0,2,4,6 列，只利用一个
    // 32-byte sector 中的 8 字节。先按 MMA 的寄存器映射写 shared，再由另一组 lane
    // 沿列连续读出，这样可以改变写回分工而不改变 MMA 的寄存器布局。
    #pragma unroll
    for (int m = 0; m < MMA_TILES_M; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_TILES_N; ++n) {
            #pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int row = warp_m + m * MMA_M + lane_id / 4 + half_row * 8;
                const int col = warp_n + n * MMA_N + (lane_id % 4) * 2;
                const float* pair = &accumulator[m][n][2 * half_row];

                // PERF: 一个 lane 拥有相邻两列，cvt 的最后一个源放在低 16 位。
                //       一次舍入、打包成一个 32-bit bank word，避免拆成两次半字写。
                uint32_t packed;
                if constexpr (std::is_same_v<T, __half>) {
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(pair[1]), "f"(pair[0]));
                } else {
                    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(packed) : "f"(pair[1]), "f"(pair[0]));
                }

                // 8 行中的同一列段轮换到不同的 4-bank 组。BN 为 32/64/128 时，
                // XOR 在每组 8 行内为双射，保留 8 元素段内顺序，读回必须用相同映射。
                const int offset = (row * BLOCK_N + col) ^ ((row & 7) << 3);
                const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(storage + offset));
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(addr), "r"(packed) : "memory");
            }
        }
    }

    // WARNING: 新的写回线程可能读取其他 warp 的结果，需等所有寄存器片段写入。
    __syncthreads();
    for (int i = threadIdx.x * 8; i < OUTPUT_SIZE; i += NUM_THREADS * 8) {
        const int row = i / BLOCK_N;
        const int col = i % BLOCK_N;
        if (row >= M - offs_m || col >= N - offs_n) continue;

        T* dst = y_tile_base + int64_t(row) * N + col;
        const T* src = storage + (i ^ ((row & 7) << 3));
        if ((reinterpret_cast<uintptr_t>(dst) & 15) == 0 && N - offs_n - col >= 8) {
            // PERF: 每线程写连续 16 字节；基准的行首按 32 字节对齐，相邻两个 lane 填满一个 sector。
            //       先读入 4 个原始字再直接存到 global，保留 FP16/BF16 已舍入的位。
            uint4 words;
            const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(words.x), "=r"(words.y), "=r"(words.z), "=r"(words.w)
                         : "r"(addr) : "memory");
            asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};"
                         :: "l"(dst), "r"(words.x), "r"(words.y), "r"(words.z), "r"(words.w) : "memory");
        } else {
            // 输出子视图也可能偏移一个 half；保持原接口，只写这一行有效的元素。
            #pragma unroll
            for (int j = 0; j < 8 && col + j < N - offs_n; ++j) dst[j] = src[j];
        }
    }

    // WARNING: persistent CTA 下一次调用会把新 X/W 写入 storage。
    //          即便某个 warp 没有有效 Y，它也必须等待其他 warp 把 shared 全部读完。
    __syncthreads();
}

}  // namespace
}  // namespace group_gemm
