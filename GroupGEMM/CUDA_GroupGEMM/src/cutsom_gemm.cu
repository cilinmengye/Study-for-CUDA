/*
 * cutsom_gemm.cu
 * 基础 SIMT Group GEMM：两个布局 kernel 共用一个块内 tile GEMM 函数。
 * 阅读顺序：tile_gemm -> 专家布局与 persistent 调度 -> host 启动入口。
 * NOTE: 输入输出为 FP16/BF16，使用 FP32 累加；W 的物理布局是 [experts,N,K]。
 */
#include "custom_gemm.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iterator>  // std::size：自动取得启动函数数组的长度。

// NOTE: 设备模板在当前翻译单元中实例化，保留原来的 __forceinline__ 调用方式。
#include "tilegemm/tilegemm_v0.cu"

namespace group_gemm {
namespace {


// ===========================
// 两种布局共用的 tile GEMM 入口
// ===========================

/*
 * x_group, w_group, y_group: [Device] 当前专家的矩阵起点。
 * M 为有效行数；offs_m/offs_n 为输出 tile 的元素坐标。
 * WARNING: 整个 CTA 必须一起调用，线程数与 NUM_THREADS 一致。
 * NOTE: 仅在此处选择版本；新增 tilegemm_v1.cu 后，包含它并将下面的转发改为 tile_gemm_v1。
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
    tile_gemm_v0<T, BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS>(
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
    for (int expert_idx = 0; expert_idx < experts; ++expert_idx) {
        const int64_t row_start = row_offsets[expert_idx];

        // get this expert the number of tile
        // 我们所谓的 tile 是指处理 (BLOCK_M, BLOCK_K)
        // NOTE: 上句描述 X 的分块；下面 num_expert_tiles 统计的是 Y 上的 (BLOCK_M,BLOCK_N) 分块。
        const int M = row_offsets[expert_idx + 1] - row_offsets[expert_idx];
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
};

template<typename T>
inline constexpr MaskedLaunchFn MaskedLaunchers[] = {
    &launch_masked_typed<T, 32, 64, 32, 128>,
    &launch_masked_typed<T, 32, 32, 16, 128>,
    &launch_masked_typed<T, 32, 64, 16, 128>,
    &launch_masked_typed<T, 64, 64, 32, 128>,
    &launch_masked_typed<T, 64, 64, 32, 256>,
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
