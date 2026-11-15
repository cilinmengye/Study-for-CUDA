/*
 * testbed.h
 * benchmark 与正确性检查共用的数据集。输入在 CPU 上量化后再上传，
 * CPU/GPU 参考值因此都使用与待测算子完全相同的低精度输入。
 */
#pragma once

#include "grouped_gemm.h"

namespace group_gemm {

/* 逻辑测试场景；rows 是有效 M 列表，capacity 仅在 masked 布局中表示物理行容量。 */
struct CaseSpec {
    std::string name;
    Layout layout;
    int n;
    int k;
    std::vector<int> rows;
    int capacity = 0;
};


/*
 * 拥有测试输入、输出和 GPU 元数据；Y 前后附加 guard 检测越界写入。
 * storage_rows 是实际分配的行数：contiguous 为 sum(rows)，masked 为 experts*capacity。
 * NOTE: masked 的分配容量通常大于有效行数，计算 FLOPs 时只能使用有效行数。
 * Dataset 的生命周期覆盖参考计算、库操作以及最后一次 Graph 执行。
 */
class Dataset {
public:
    static constexpr size_t kGuardElements = 64;

    Dataset(CaseSpec spec, DType dtype, cudaStream_t stream);
    GroupedProblem problem() const;
    void reset_output();
    std::vector<uint16_t> download_output() const;

    CaseSpec spec;
    DType dtype;
    cudaStream_t stream;
    int64_t storage_rows;
    // [Host] 已量化为 FP16/BF16 后再转换回 float 的精确输入值，只用于参考计算。
    std::vector<float> host_a;
    std::vector<float> host_w;
    // [Device] a/w/y_with_guards 实际保存 16-bit 元素，并不是上面 host vector 的 float 格式。
    DeviceBuffer a;
    DeviceBuffer w;
    DeviceBuffer y_with_guards;
    // [Device] int32；两种布局分别保存行前缀和、各专家有效行数。
    DeviceBuffer metadata;
    uint16_t sentinel;
};


/* 有效输出的误差统计；任何数值或 guard 错误会直接抛异常，不以此结构返回“部分成功”。 */
struct CheckResult {
    double max_abs_error = 0;
    size_t checked_elements = 0;
};

std::vector<float> make_reference(const Dataset& dataset, bool full_cpu);
CheckResult check_output(const Dataset& dataset, const std::vector<float>& reference);
CheckResult validate(GemmOperation& operation, Dataset& dataset, const std::vector<float>& reference);
std::vector<CaseSpec> primary_cases();
std::vector<CaseSpec> edge_cases();

}  // namespace group_gemm
