/*
 * cutsom_gemm.cu
 * Tensor Core Group GEMM：两个布局 kernel 共用一个块内 tile GEMM 函数。
 * 阅读顺序：tile_gemm -> 专家布局与 persistent 调度 -> host 启动入口。
 * NOTE: 输入输出为 FP16/BF16，使用 FP32 累加；W 的物理布局是 [experts,N,K]。
 */
#include "custom_gemm.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iterator>  // std::size：自动取得启动函数数组的长度。

// NOTE: 设备模板在当前翻译单元中实例化，保留原来的 __forceinline__ 调用方式。
#include "tilegemm/tilegemm_v0.cu"
#include "tilegemm/tilegemm_v1_TensorCore_Swizzle.cu"
#include "tilegemm/tilegemm_v3_Merge_MemoryAccess.cu"

namespace group_gemm {
namespace {


// ===========================
// 两种布局共用的 tile GEMM 入口
// ===========================

/*
 * x_group, w_group, y_group: [Device] 当前专家的矩阵起点。
 * M 为有效行数；offs_m/offs_n 为输出 tile 的元素坐标。
 * WARNING: 整个 CTA 必须一起调用，线程数与 NUM_THREADS 一致。
 * NOTE: v3 复用 v2 的异步双缓冲，并重排 Tensor Core 结果以连续写回。
 *       改回 tile_gemm_v1 即可对照同步 Tensor Core 实现，v0 保留为 SIMT 基线。
 */
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__device__ __forceinline__ void tile_gemm(
    const T* x_group,
    const T* w_group,
    T* y_group,
    int M,
    int N,
    int K,
    int offs_m,
    int offs_n
) {
    tile_gemm_v3<T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS>(
        x_group, w_group, y_group, M, N, K, offs_m, offs_n
    );
}



// ===========================
// 专家布局与 persistent tile 调度
// ===========================

/*
 * x, w, y: [Device]，元素类型为 T。
 * row_offsets: [Device]，experts + 1 个行偏移。
 * NOTE: T 决定输入输出存储类型，累加仍使用 float。
 * x, w, y: [Device] 紧凑 X/Y 与逐专家权重。
 * row_offsets: [Device] experts+1 个行偏移，允许重复偏移表示空专家。
 * NOTE: block 领取全局 tile 编号，处理完后按 gridDim.x 跨步继续。
 */
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) custom_contiguous_group_gemm_kernel(
    const T* x,
    const T* w,
    T* y,
    const int32_t* row_offsets,
    int experts,
    int N,
    int K
) {
    int64_t tile_idx = blockIdx.x;
    int64_t start_expert_tile_range = 0;
    const int64_t tiles_n = (int64_t(N) + BLOCK_N - 1) / BLOCK_N;

    // Group GEMM 开始遍历 Group, 对每个 Group 执行 GEMM
    // 每个专家的逻辑行数与物理起点都由行偏移描述。
    int64_t row_start = 0;
    for (int expert_idx = 0; expert_idx < experts; ++expert_idx) {
        const int64_t row_end = row_offsets[expert_idx + 1];

        // get this expert the number of tile
        // 我们所谓的 tile 是指处理 (BLOCK_M, BLOCK_K)
        // NOTE: 上句描述 X 的分块；下面 num_expert_tiles 统计的是 Y 上的 (BLOCK_M,BLOCK_N) 分块。
        const int M = static_cast<int>(row_end - row_start);
        const int64_t num_expert_tiles = ((int64_t(M) + BLOCK_M - 1) / BLOCK_M) * tiles_n;
        const int64_t end_expert_tile_range = start_expert_tile_range + num_expert_tiles;

        // 这种 persistent GEMM 处理方式能够天然地处理 num_expert_tile = 0
        // 即 M = 0 时空输入的情况
        // 当空输入时必然会出现 tile_idx >= end_expert_tile_range
        // 然后我们会跳过这个 expert GEMM 计算
        // NOTE: tile_idx 单调递增，始终不小于当前区间起点；空专家自然跳过。
        while (tile_idx < end_expert_tile_range) {
            const int64_t local_tile = tile_idx - start_expert_tile_range;
            const int offs_m = int(local_tile / tiles_n) * BLOCK_M;
            const int offs_n = int(local_tile % tiles_n) * BLOCK_N;

            // 每个 x,w,y group 的 base address ptr
            // NOTE: 直接在下面的实参中计算；W 由 expert_idx 定位，X/Y 由 row_start 定位。
            // 整个 CTA 使用相同的专家与 tile 坐标，共同进入含 barrier 的计算核心。
            // 传入的 x,w,y 都是已经移动到了对应 group 的起始地址上。
            tile_gemm<T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS>(
                x + row_start * K,
                w + int64_t(expert_idx) * N * K,
                y + row_start * N,
                M, N, K, offs_m, offs_n
            );

            tile_idx += gridDim.x;
        }

        start_expert_tile_range = end_expert_tile_range;
        row_start = row_end;
    }
}


/*
 * x, w, y: [Device] X/Y 按专家预留固定 capacity，W 仍为 [experts,N,K]。
 * valid_rows: [Device] 每个专家的有效行数，范围为 [0,capacity]。
 * NOTE: 物理起点为 expert*capacity；只为 valid_rows 对应的有效输出分配 tile。
 */
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) custom_masked_group_gemm_kernel(
    const T* x,
    const T* w,
    T* y,
    const int32_t* valid_rows,
    int experts,
    int capacity,
    int N,
    int K
) {
    int64_t tile_idx = blockIdx.x;
    int64_t start_expert_tile_range = 0;
    const int64_t tiles_n = (int64_t(N) + BLOCK_N - 1) / BLOCK_N;

    // 有效 M 决定工作量，capacity 决定物理步长，两者不能混用。
    for (int expert_idx = 0; expert_idx < experts; ++expert_idx) {
        const int64_t row_start = int64_t(expert_idx) * capacity;
        const int M = valid_rows[expert_idx];
        const int64_t num_expert_tiles = ((int64_t(M) + BLOCK_M - 1) / BLOCK_M) * tiles_n;
        const int64_t end_expert_tile_range = start_expert_tile_range + num_expert_tiles;

        // NOTE: 与 contiguous 相同的 tile 调度；零有效行不会进入计算核心。
        while (tile_idx < end_expert_tile_range) {
            const int64_t local_tile = tile_idx - start_expert_tile_range;
            const int offs_m = int(local_tile / tiles_n) * BLOCK_M;
            const int offs_n = int(local_tile % tiles_n) * BLOCK_N;

            // 核心只接收专家矩阵起点与有效 M，无需知道它来自哪种物理布局。
            tile_gemm<T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS>(
                x + row_start * K,
                w + int64_t(expert_idx) * N * K,
                y + row_start * N,
                M, N, K, offs_m, offs_n
            );

            tile_idx += gridDim.x;
        }

        start_expert_tile_range = end_expert_tile_range;
    }
}



// ===========================
// Host 启动与类型分派
// ===========================

/* grid_blocks: [Host] 输出当前设备的 SM 数，作为基础 persistent grid 的 CTA 数量。
 * NOTE: 这只决定 CTA 数量，不保证每个 CTA 固定绑定某个 SM。
 */
cudaError_t get_grid_blocks(int& grid_blocks) {
    int device = 0;
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    return cudaDeviceGetAttribute(&grid_blocks, cudaDevAttrMultiProcessorCount, device);
}


// `get_tuned_grid_blocks`代码目的: 提高 Theoretical Occupancy，是为了让 SM 能驻留更多独立的 warp：
// 当一部分 warp 等待数据或同步时，调度器还能选择其他 warp 执行。最终目标是减少发射空档、缩短 kernel 时间。
// Theoretical Occupancy = 当前 kernel 配置允许驻留的 warp 数 / 硬件最大驻留 warp 数
// 当前 kernel 配置允许驻留的 warp 数受到 warp 消耗多少资源限制。
// 也可以理解为：Theoretical Occupancy 主要受每个 SM 的资源容量，以及每个 CTA 消耗多少资源限制。
// NOTE: 实际采用的数量不会超过 occupancy 查询上限。
constexpr int TARGET_CTAS_PER_SM = 64;

// NOTE: -1 表示让 CUDA 自动选择。
//       0..100 表示共享内存容量偏好的百分比，不是 KiB。
constexpr int SHARED_CARVEOUT_PERCENT = cudaSharedmemCarveoutDefault;

/*
 * 用来查询 Kernel 实际 SM 能够存放下多少个 CTA.
 * grid_blocks: [Host] 输出最终 grid 大小。
 * sm_count: 当前设备的 SM 数量。
 *
 * NOTE: Kernel 是编译期 kernel 符号，因此每个具体 kernel 有独立缓存。
 *       在 Graph capture 前先正常调用一次，完成初始化；
 *       现有 benchmark 的预热流程已经满足这一点。
 *       此简化缓存假定进程内不调用 cudaDeviceReset，
 *       也不在其他代码中更改这些 kernel 属性。
 */
template<auto Kernel, int NUM_THREADS>
cudaError_t get_tuned_grid_blocks(int sm_count, int& grid_blocks) {
    static_assert(TARGET_CTAS_PER_SM > 0, "CTA target must be positive");
    static_assert(
        SHARED_CARVEOUT_PERCENT >= -1 && SHARED_CARVEOUT_PERCENT <= 100,
        "Shared-memory preference must be -1 or a percentage"
    );

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }

    // PERF: 避免每次 launch 都重新设置属性、查询 occupancy。
    //       每个 CPU 线程缓存最近使用的设备；切换设备时重新查询。
    static thread_local int cached_device = -1;
    static thread_local int cached_ctas_per_sm = 0;

    if (cached_device != device) {
        // 先设置当前 kernel 的共享内存偏好。
        status = cudaFuncSetAttribute(
            Kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            SHARED_CARVEOUT_PERCENT
        );
        if (status != cudaSuccess) {
            return status;
        }

        // 查询资源允许的最大驻留 CTA 数。
        // NOTE: 0 表示动态共享内存为零；静态 __shared__ 数组会自动计入。
        int max_ctas_per_sm = 0;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_ctas_per_sm, Kernel, NUM_THREADS, 0
        );
        if (status != cudaSuccess) {
            return status;
        }
        if (max_ctas_per_sm == 0) {
            return cudaErrorInvalidConfiguration;
        }

        // 例如目标为 4、资源上限为 2，则实际按 2 计算。
        cached_ctas_per_sm = TARGET_CTAS_PER_SM < max_ctas_per_sm
            ? TARGET_CTAS_PER_SM : max_ctas_per_sm;
        cached_device = device;
    }

    // NOTE: 这里只决定全卡 CTA 数量，具体分配到哪个 SM 仍由 GPU 调度。
    grid_blocks = sm_count * cached_ctas_per_sm;
    return cudaSuccess;
}


/* args: [Host] 参数结构，所有指针均为 [Device]；仅向调用者的 stream 提交计算。 */
template<
    typename T,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int NUM_THREADS
>
cudaError_t launch_contiguous_typed(
    const ContiguousGemmArgs& args,
    cudaStream_t stream,
    int grid_blocks
) {
    // 外层传入的 grid_blocks 当前其实是 SM 数量。
    const int sm_count = grid_blocks;

    const cudaError_t status = get_tuned_grid_blocks<
        custom_contiguous_group_gemm_kernel<
            T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS
        >,
        NUM_THREADS
    >(sm_count, grid_blocks);

    if (status != cudaSuccess) {
        return status;
    }

    custom_contiguous_group_gemm_kernel<
        T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS
    ><<<grid_blocks, NUM_THREADS, 0, stream>>>(
        static_cast<const T*>(args.x),
        static_cast<const T*>(args.w),
        static_cast<T*>(args.y),
        args.row_offsets,
        args.experts,
        args.n,
        args.k
    );

    return cudaGetLastError();
}



/* args: [Host] 参数结构，所有指针均为 [Device]；capacity 作为物理行步长传入。 */
template<
    typename T,
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int NUM_THREADS
>
cudaError_t launch_masked_typed(
    const MaskedGemmArgs& args,
    cudaStream_t stream,
    int grid_blocks
) {
    const int sm_count = grid_blocks;

    const cudaError_t status = get_tuned_grid_blocks<
        custom_masked_group_gemm_kernel<
            T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS
        >,
        NUM_THREADS
    >(sm_count, grid_blocks);

    if (status != cudaSuccess) {
        return status;
    }

    custom_masked_group_gemm_kernel<
        T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS
    ><<<grid_blocks, NUM_THREADS, 0, stream>>>(
        static_cast<const T*>(args.x),
        static_cast<const T*>(args.w),
        static_cast<T*>(args.y),
        args.valid_rows,
        args.experts,
        args.capacity,
        args.n,
        args.k
    );

    return cudaGetLastError();
}


// ================ 用于自动调优 ========================
// NOTE: 以下都是 Host 函数指针；args 内保存的是 Device 地址。
using ContiguousLaunchFn = cudaError_t (*)(
    const ContiguousGemmArgs&,
    cudaStream_t,
    int
);

using MaskedLaunchFn = cudaError_t (*)(
    const MaskedGemmArgs&,
    cudaStream_t,
    int
);

template<typename T>
inline constexpr ContiguousLaunchFn ContiguousLaunchers[] = {
    // BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS
    &launch_contiguous_typed<T, 32, 64, 32, 128>,
    &launch_contiguous_typed<T, 32, 32, 16, 128>,
    &launch_contiguous_typed<T, 32, 64, 16, 128>,
    &launch_contiguous_typed<T, 64, 64, 32, 128>,
    &launch_contiguous_typed<T, 64, 64, 32, 256>,
    // v3：先在同一 64x64 tile 上比较 BK=32/64，再考察 BN 翻倍后的 X 复用。
    &launch_contiguous_typed<T, 64, 64, 64, 128>,
    &launch_contiguous_typed<T, 64, 128, 32, 128>,
    &launch_contiguous_typed<T, 64, 128, 64, 128>,
    &launch_contiguous_typed<T, 128, 128, 32, 128>
};

template<typename T>
inline constexpr MaskedLaunchFn MaskedLaunchers[] = {
    &launch_masked_typed<T, 16, 64, 32, 128>,
    &launch_masked_typed<T, 16, 32, 16, 128>,
    &launch_masked_typed<T, 16, 64, 16, 128>,
    &launch_masked_typed<T, 32, 64, 32, 128>,
    &launch_masked_typed<T, 32, 32, 16, 128>,
    &launch_masked_typed<T, 32, 64, 16, 128>,
    &launch_masked_typed<T, 64, 64, 32, 128>,
    // Decode 的专家有效行较少，保留窄 BM；扩大 BK 使每行的 8 个 lane 搬满 128 字节。
    &launch_masked_typed<T, 16, 64, 64, 128>,
    &launch_masked_typed<T, 32, 64, 64, 128>,
    &launch_masked_typed<T, 16, 128, 64, 128>
};

}  // namespace


int contiguous_config_count() {
    return static_cast<int>(std::size(ContiguousLaunchers<__half>));
}


int masked_config_count() {
    return static_cast<int>(std::size(MaskedLaunchers<__half>));
}


// ========================== 调用接口 ===================================

/* args: [Host] 参数结构，所有指针均为 [Device]；空专家集合无需访问任何指针。 */
cudaError_t launch_contiguous(
    const ContiguousGemmArgs& args,
    cudaStream_t stream,
    int config
) {
    // 在访问函数指针数组前检查配置编号。
    if (config < 0 || config >= contiguous_config_count()) {
        return cudaErrorInvalidValue;
    }

    if (args.experts == 0) {
        return cudaSuccess;
    }

    // 从当前设备获得启动规模，不在入口中分配、拷贝或同步 GPU 数据。
    int grid_blocks = 0;
    const cudaError_t status = get_grid_blocks(grid_blocks);
    if (status != cudaSuccess) {
        return status;
    }

    // dtype 选择类型对应的表，config 选择表中的配置。
    switch (args.dtype) {
        case DType::FP16:
            return ContiguousLaunchers<__half>[config](
                args, stream, grid_blocks);
        case DType::BF16:
            return ContiguousLaunchers<__nv_bfloat16>[config](
                args, stream, grid_blocks);
        default:
            return cudaErrorInvalidValue;
    }
}


/* args: [Host] 参数结构，所有指针均为 [Device]；GPU 执行时读取 valid_rows。 */
cudaError_t launch_masked(
    const MaskedGemmArgs& args,
    cudaStream_t stream,
    int config
) {
    // 在访问函数指针数组前检查配置编号。
    if (config < 0 || config >= masked_config_count()) {
        return cudaErrorInvalidValue;
    }

    if (args.experts == 0) {
        return cudaSuccess;
    }

    // 与 contiguous 使用相同的基础 grid 和 tile 配置。
    int grid_blocks = 0;
    const cudaError_t status = get_grid_blocks(grid_blocks);
    if (status != cudaSuccess) {
        return status;
    }

    // dtype 选择类型对应的表，config 选择表中的配置。
    switch (args.dtype) {
        case DType::FP16:
            return MaskedLaunchers<__half>[config](
                args, stream, grid_blocks);
        case DType::BF16:
            return MaskedLaunchers<__nv_bfloat16>[config](
                args, stream, grid_blocks);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace group_gemm
