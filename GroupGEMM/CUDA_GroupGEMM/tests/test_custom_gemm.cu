/*
 * test_custom_gemm.cu
 * 通过公共启动接口验证共享 tile 核心、两种布局及 Graph 中的设备元数据读取。
 * 小矩阵使用现有 CPU FP64 全量参考，并检查 masked padding 与输出 guard。
 * 每个布局使用自己的配置数量，逐个验证所有候选及其 Graph 回放。
 * NOTE: 此目标规模较小，也用于 Compute Sanitizer；不据此报告性能。
 */
#include "custom_gemm.h"
#include "testbed.h"

#include <iostream>

namespace group_gemm {
namespace {

/* 独占一条非默认 stream，确保测试不依赖 legacy default stream 的隐式同步。 */
class TestStream {
public:
    TestStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }


    ~TestStream() { cudaStreamDestroy(value); }

    TestStream(const TestStream&) = delete;
    TestStream& operator=(const TestStream&) = delete;
    cudaStream_t value = nullptr;
};


/* dataset: [Host] 测试数据集，problem() 内所有矩阵及元数据指针为 [Device]。 */
void launch_dataset(const Dataset& dataset, int config) {
    const auto p = dataset.problem();
    if (p.layout == Layout::Contiguous) {
        // 使用与 benchmark 相同的公共入口。
        const ContiguousGemmArgs args{
            p.a, p.w, p.y, p.metadata, p.experts(), p.n, p.k, p.dtype
        };
        CUDA_CHECK(launch_contiguous(args, dataset.stream, config));
    } else {
        // 传递物理容量，有效行数仍从设备元数据读取。
        const MaskedGemmArgs args{
            p.a, p.w, p.y, p.metadata, p.experts(), p.capacity, p.n, p.k, p.dtype
        };
        CUDA_CHECK(launch_masked(args, dataset.stream, config));
    }
}


/* 拥有 Graph 及 executable；借用 dataset 的设备内存，必须先于 dataset 销毁。 */
class Replay {
public:
    explicit Replay(const Dataset& dataset, int config) {
        // 捕获两次调用，验证多次覆写输出与共享内存复用。
        // NOTE: 捕获当前候选，避免 Graph 路径意外退回默认配置 0。
        CUDA_CHECK(cudaStreamBeginCapture(dataset.stream, cudaStreamCaptureModeThreadLocal));
        launch_dataset(dataset, config);
        launch_dataset(dataset, config);
        CUDA_CHECK(cudaStreamEndCapture(dataset.stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable_, graph_, 0));
    }


    ~Replay() {
        cudaGraphExecDestroy(executable_);
        cudaGraphDestroy(graph_);
    }

    Replay(const Replay&) = delete;
    Replay& operator=(const Replay&) = delete;


    void run(cudaStream_t stream) { CUDA_CHECK(cudaGraphLaunch(executable_, stream)); }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};


void check_case(const CaseSpec& spec, DType dtype, cudaStream_t stream, int config) {
    Dataset dataset(spec, dtype, stream);
    // 将配置编号加入错误上下文，定位失败候选而不改变测试数据。
    dataset.spec.name += "/config=" + std::to_string(config);
    const auto reference = make_reference(dataset, true);

    // 直接提交与 Graph 回放都通过独立数值参考和 padding/guard 检查。
    launch_dataset(dataset, config);
    check_output(dataset, reference);
    Replay replay(dataset, config);
    dataset.reset_output();
    replay.run(stream);
    check_output(dataset, reference);

    if (spec.layout == Layout::Masked) {
        // NOTE: 只缩小有效范围，保留原来的输入、容量、地址与 Graph。
        //       若 kernel 固化了旧 M，重新设为 sentinel 的无效输出会被错误覆写。
        for (size_t e = 0; e < dataset.spec.rows.size(); ++e) {
            dataset.spec.rows[e] = e % 2 == 0 ? 0 : dataset.spec.rows[e] / 2;
        }
        std::vector<int32_t> rows(dataset.spec.rows.begin(), dataset.spec.rows.end());
        dataset.metadata.upload(rows.data(), stream);
        dataset.reset_output();
        replay.run(stream);
        check_output(dataset, make_reference(dataset, true));

        // 恢复初始有效范围，验证同一 Graph 能重新处理从零行恢复的专家。
        // NOTE: 初始有效输入一直保留在原缓冲区中，不读取原先的 NaN padding。
        dataset.spec.rows = spec.rows;
        rows.assign(spec.rows.begin(), spec.rows.end());
        dataset.metadata.upload(rows.data(), stream);
        dataset.reset_output();
        replay.run(stream);
        check_output(dataset, reference);
    }

    std::cout << "PASS " << dtype_name(dtype) << '/' << layout_name(spec.layout)
              << '/' << spec.name << " config=" << config << " normal+graph\n";
}


void check_no_experts(DType dtype, cudaStream_t stream) {
    // 空集合必须允许空指针，不能访问矩阵或元数据。
    const ContiguousGemmArgs contiguous{nullptr, nullptr, nullptr, nullptr, 0, 8, 8, dtype};
    const MaskedGemmArgs masked{nullptr, nullptr, nullptr, nullptr, 0, 8, 8, 8, dtype};
    for (int config = 0; config < contiguous_config_count(); ++config) {
        CUDA_CHECK(launch_contiguous(contiguous, stream, config));
    }
    for (int config = 0; config < masked_config_count(); ++config) {
        CUDA_CHECK(launch_masked(masked, stream, config));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 当前入口先检查配置编号；即使专家集合为空，也必须拒绝非法数组下标。
    for (int config : {-1, contiguous_config_count()}) {
        if (launch_contiguous(contiguous, stream, config) != cudaErrorInvalidValue) {
            throw std::runtime_error("contiguous accepted invalid config=" + std::to_string(config));
        }
    }
    for (int config : {-1, masked_config_count()}) {
        if (launch_masked(masked, stream, config) != cudaErrorInvalidValue) {
            throw std::runtime_error("masked accepted invalid config=" + std::to_string(config));
        }
    }
    std::cout << "PASS " << dtype_name(dtype) << "/no_experts+config_bounds\n";
}

}  // namespace
}  // namespace group_gemm


int main() {
    using namespace group_gemm;
    try {
        // 让小尺寸测试也有多于 grid 大小的 tile，覆盖跨专家的 persistent 复用。
        TestStream stream;
        int sm_count = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

        // 两个布局可以有不同的候选数量，禁止空表造成测试静默跳过。
        const int contiguous_count = contiguous_config_count();
        const int masked_count = masked_config_count();
        if (contiguous_count <= 0 || masked_count <= 0) {
            throw std::runtime_error("Both layouts must provide at least one configuration");
        }
        std::cout << "Configurations: contiguous=" << contiguous_count
                  << ", masked=" << masked_count << '\n';

        auto cases = edge_cases();
        for (Layout layout : {Layout::Contiguous, Layout::Masked}) {
            cases.push_back({"mnk_tail", layout, 72, 40, {0, 33, 0, 65, 1}, 80});
            cases.push_back({"persistent", layout, 8, 8, std::vector<int>(sm_count + 3, 1), 3});
        }

        // 每个候选使用独立数据集与 Graph，避免上个候选的输出掩盖漏写。
        int checks = 0;
        for (DType dtype : {DType::FP16, DType::BF16}) {
            check_no_experts(dtype, stream.value);
            for (const auto& spec : cases) {
                const int config_count = spec.layout == Layout::Contiguous
                    ? contiguous_count : masked_count;
                for (int config = 0; config < config_count; ++config) {
                    check_case(spec, dtype, stream.value, config);
                    ++checks;
                }
            }
        }
        std::cout << "All custom GEMM checks passed: " << checks << " case/config combinations.\n";
        return 0;
    } catch (const std::exception& error) {
        // 失败时保留具体案例和 CUDA/数值检查错误。
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
