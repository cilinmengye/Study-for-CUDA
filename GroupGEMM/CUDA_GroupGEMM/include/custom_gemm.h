/*
 * custom_gemm.h
 * 自定义 contiguous/masked CUDA 入口；此文件只定义接口，不实现计算 kernel。
 * 实现在 src/cutsom_gemm.cu，通过 -DENABLE_CUSTOM=ON 接入测试。
 */
#pragma once

#include "grouped_gemm.h"

namespace group_gemm {

/*
 * a: [Device] [row_offsets[experts], k]，按专家顺序紧凑拼接。
 * w: [Device] [experts, n, k]，每个专家有独立权重。
 * y: [Device] [row_offsets[experts], n]，由调用者预分配。
 * row_offsets: [Device] experts+1 个 int32，首元素为 0，允许重复偏移（空专家）。
 * dtype: FP16/BF16 输入输出，计算必须使用 FP32 累加。
 *
 * 专家 e 的逻辑行数 = row_offsets[e+1] - row_offsets[e]。
 * 若把指针转换为对应的 16-bit 元素类型，元素地址为：
 *   A_e[m,k_idx] = a[(row_offsets[e] + m) * k + k_idx]
 *   W_e[n_idx,k_idx] = w[(e * n + n_idx) * k + k_idx]
 *   Y_e[m,n_idx] = y[(row_offsets[e] + m) * n + n_idx]
 * NOTE: 这些是 GPU kernel 中的索引公式，CPU 不能直接读取 device row_offsets[e]。
 */
struct ContiguousGemmArgs {
    const void* x;
    const void* w;
    void* y;
    const int32_t* row_offsets;
    int experts;
    int n;
    int k;
    DType dtype;
};


/*
 * a: [Device] [experts, capacity, k]，仅各专家前 valid_rows[e] 行有效。
 * w: [Device] [experts, n, k]。
 * y: [Device] [experts, capacity, n]，无效行必须保持原值。
 * valid_rows: [Device] experts 个 int32，取值范围 [0, capacity]。
 * NOTE: 无效 A 行可能为 NaN，不能让它们污染有效输出。
 *
 * 与 contiguous 不同，专家 e 始终从第 e*capacity 个物理行开始：
 *   A_e[m,k_idx] = a[(e * capacity + m) * k + k_idx]
 *   Y_e[m,n_idx] = y[(e * capacity + m) * n + n_idx]
 * 有效条件是 0 <= m < valid_rows[e]；m >= valid_rows[e] 时不能写 Y。
 * W 的索引方式与 contiguous 完全相同。这里的 mask 是每组有效行数，不是任意布尔矩阵。
 */
struct MaskedGemmArgs {
    const void* x;
    const void* w;
    void* y;
    const int32_t* valid_rows;
    int experts;
    int capacity;
    int n;
    int k;
    DType dtype;
};


// 返回候选数量，供外部验证和调优循环使用。
int contiguous_config_count();
int masked_config_count();

/* args 内所有指针均为 [Device]；仅在 stream 中启动计算并返回启动状态。
 * NOTE: 不在入口里分配、同步或拷贝元数据；须能被 CUDA Graph 捕获。
 *       n、k 为正的 8 的倍数，M 可以不对齐或为 0，所有矩阵按行存储。
 * WARNING: args 是 host 上的参数结构，但里面保存的是 device 地址。
 *          要按 dtype 将 void* 转为 half/bfloat16 指针后再传给自己的 kernel。
 *          FP32 累加不代表输出缓冲区是 float：最后仍需写回对应的 16-bit 类型。
 */
// NOTE: config 是对应启动函数数组的下标；默认 0 保持原来的配置。
cudaError_t launch_contiguous(
    const ContiguousGemmArgs& args,
    cudaStream_t stream,
    int config = 0
);

cudaError_t launch_masked(
    const MaskedGemmArgs& args,
    cudaStream_t stream,
    int config = 0
);

}  // namespace group_gemm
