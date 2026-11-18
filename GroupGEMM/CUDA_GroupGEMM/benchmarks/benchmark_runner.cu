/*
 * benchmark_runner.cu
 * 边界验证、候选选优、正式测量和 CSV 输出；原有流程与函数体保持不变。
 */
#include "benchmark_runner.h"
#include "benchmark_timing.h"
#include "custom_operation.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <nvtx3/nvToolsExt.h>

namespace group_gemm {

void check_edges(const Options& options, DType dtype, cudaStream_t stream) {
    // NOTE: 此处是测试流程的调用入口，数据和断言由 tests/test_correctness.cu 提供。
    //       即使只选择一个主性能场景，也先执行完整的小尺寸边界检查。
    int checks = 0;
    for (const auto& spec : edge_cases()) {
        Dataset dataset(spec, dtype, stream);
        const auto reference = make_reference(dataset, true);
        if (wants(options, "cublas")) {
            // 包含空专家和全空集合，不只验证主性能场景。
            auto operation = make_cublas(dataset.problem(), stream);
            validate(*operation, dataset, reference);
            ++checks;
        }
        if (wants(options, "cutlass")) {
            // 每个候选都要通过 CPU FP64 参考与 padding/guard 检查。
            for (int config = 0; config < kCutlassConfigs; ++config) {
                auto operation = make_cutlass(dataset.problem(), config, stream);
                validate(*operation, dataset, reference);
                ++checks;
            }
        }
#ifdef ENABLE_CUSTOM
        if (wants(options, "custom")) {
            // 每个候选都验证边界案例，两个布局分别查询候选数量。
            const int config_count = spec.layout == Layout::Contiguous
                ? contiguous_config_count() : masked_config_count();

            for (int config = 0; config < config_count; ++config) {
                CustomOperation operation(dataset.problem(), stream, config);
                validate(operation, dataset, reference);
                ++checks;
            }
        }
#endif
    }
    std::cout << "CPU FP64 边界检查通过: " << dtype_name(dtype) << ", " << checks << " 次算子检查\n";
}


namespace {

int tune_candidates(std::vector<OperationPtr>& candidates, Dataset& dataset,
                 const std::vector<float>& reference, const Options& options) {
    // NOTE: 调用者已经验证全部候选的单次输出；这里还检查实际 Graph 回放的输出。
    //       timers 借用 candidates 中的对象，函数退出时先销毁 Graph，才允许调用者释放候选。
    std::vector<std::unique_ptr<BatchTimer>> timers;
    for (auto& candidate : candidates) {
        // 在捕获之前预热，排除延迟加载与初次调用的影响。
        warm_up(*candidate, dataset.stream, options.warmup);
        timers.push_back(std::make_unique<BatchTimer>(*candidate, true, options.iterations, dataset.stream));
        dataset.reset_output();
        timers.back()->launch_batch();
        check_output(dataset, reference);
    }

    const auto samples = measure_rounds(timers, 5);
    // 这五轮只用于选配置，不能直接当作最终成绩，否则会把“选得最快”的采样偏差带入报告。
    // 选中的同一配置稍后分别参加 normal 和 graph 的正式测量，不针对两种模式换不同配置。
    int best = 0;
    for (int i = 0; i < int(candidates.size()); ++i) {
        std::cout << "  候选 " << i << ": " << candidates[i]->config()
                  << " graph=" << median(samples[i]) << " us\n";
        if (median(samples[i]) < median(samples[best])) {
            best = i;
        }
    }
    std::cout << "  选择候选 " << best << "；正式计时重新采样。\n";
    return best;
}


/* 将后端名称、操作对象和正确性结果一起交给正式计时流程。 */
struct NamedOperation {
    std::string provider;
    OperationPtr operation;
    double max_abs_error;
};

}  // namespace



// ===========================
// 单个场景：准备、验证、选配置、正式测量
// ===========================

void run_case(const CaseSpec& spec, DType dtype, const Options& options,
              cudaStream_t stream, const Environment& environment, std::ofstream& csv) {
    // NOTE: dataset 先构造、最后析构，确保 operations 和 Graph 引用的 GPU 内存始终有效。
    //       构造数据集、生成参考和建立后端对象都不是 GEMM 计算本身，因此不放入 event 区间。
    Dataset dataset(spec, dtype, stream);
    const auto reference = make_reference(dataset, false);
    const auto problem = dataset.problem();
    std::vector<NamedOperation> operations;
    std::cout << "\n" << spec.name << " / " << dtype_name(dtype)
              << " / " << layout_name(spec.layout) << " / FLOPs=" << problem.flops() << '\n';

    if (wants(options, "cublas")) {
        // 所有基线先验证，不能将数值错误的运行作为快配置。
        auto operation = make_cublas(problem, stream);
        const auto result = validate(*operation, dataset, reference);
        operations.push_back({"cublas", std::move(operation), result.max_abs_error});
        std::cout << "  cuBLAS 正确性通过，max_abs=" << result.max_abs_error << '\n';
    }
    if (wants(options, "cutlass")) {
        // 保留候选对象直到筛选结束，以保证 Graph 引用的元数据有效。
        std::vector<OperationPtr> candidates;
        std::vector<CheckResult> checks;
        for (int config = 0; config < kCutlassConfigs; ++config) {
            candidates.push_back(make_cutlass(problem, config, stream));
            checks.push_back(validate(*candidates.back(), dataset, reference));
            std::cout << "  CUTLASS " << config << " 正确性通过，max_abs=" << checks.back().max_abs_error << '\n';
        }
        if (!options.check_only) {
            // 先对全部候选验证，再使用 Graph 比较计算部分的表现。
            const int best = tune_candidates(candidates, dataset, reference, options);
            operations.push_back({"cutlass", std::move(candidates[best]), checks[best].max_abs_error});
        }
    }
#ifdef ENABLE_CUSTOM
    if (wants(options, "custom")) {
        // 先验证全部候选，再在准备阶段选择配置。
        const int config_count = problem.layout == Layout::Contiguous
            ? contiguous_config_count() : masked_config_count();

        if (options.custom_config >= config_count) {
            throw std::runtime_error("--custom-config out of range for " + std::string(layout_name(problem.layout)));
        }
        const int first = options.custom_config < 0 ? 0 : options.custom_config;
        const int end = options.custom_config < 0 ? config_count : first + 1;

        std::vector<OperationPtr> candidates;
        std::vector<CheckResult> checks;

        for (int config = first; config < end; ++config) {
            candidates.push_back(
                std::make_unique<CustomOperation>(problem, stream, config));
            checks.push_back(validate(*candidates.back(), dataset, reference));

            std::cout << "  custom " << config
                      << " 正确性通过，max_abs="
                      << checks.back().max_abs_error << '\n';
        }

        if (!options.check_only) {
            // 计时器销毁后再移动获胜对象，正式测量使用同一配置。
            // 固定配置时 candidates 仅含该候选，索引为零。
            const int best = options.custom_config < 0
                ? tune_candidates(candidates, dataset, reference, options) : 0;

            operations.push_back({
                "custom",
                std::move(candidates[best]),
                checks[best].max_abs_error
            });
        }
    }
#endif
    if (options.check_only) {
        // 正确性命令不进行配置筛选，不生成性能 CSV。
        return;
    }

    for (const std::string mode : {"normal", "graph"}) {
        if (options.timing != "both" && options.timing != mode) {
            continue;
        }

        std::vector<std::unique_ptr<BatchTimer>> timers;
        for (auto& entry : operations) {
            // 所有初始化、Graph 构建和结果验证均位于 event 计时之外。
            // 两库在同一 dataset 上操作，使用相同输入和输出容量；beta=0 允许复用 Y。
            warm_up(*entry.operation, stream, options.warmup);
            timers.push_back(std::make_unique<BatchTimer>(*entry.operation, mode == "graph", options.iterations, stream));
            dataset.reset_output();
            timers.back()->launch_batch();
            check_output(dataset, reference);
        }

        // NOTE: 只标记正式测量；候选调优也会调用 measure_rounds。
        nvtxRangePushA("formal");
        const auto samples = measure_rounds(timers, options.repeats);
        nvtxRangePop();

        for (size_t i = 0; i < operations.size(); ++i) {
            const double latency_us = median(samples[i]);
            // NOTE: TFLOPS = FLOPs / 秒 / 10^12，latency_us 换成秒需要乘 10^-6，
            //       合并后分母就是 latency_us * 10^6。FLOPs 来自有效 M，而不是 masked capacity。
            const double tflops = double(problem.flops()) / (latency_us * 1e6);
            const auto& entry = operations[i];
            std::cout << "  " << std::setw(7) << entry.provider << " " << std::setw(6) << mode
                      << "  " << std::fixed << std::setprecision(3) << latency_us << " us  "
                      << tflops << " TFLOPS  PASS\n" << std::defaultfloat << std::setprecision(6);

            // CSV 写文件位于计时结束之后；quoted() 保护配置描述/GPU 名称中的空格或分隔字符。
            csv << spec.name << ',' << layout_name(spec.layout) << ',' << dtype_name(dtype) << ','
                << entry.provider << ',' << mode << ',' << problem.n << ',' << problem.k << ','
                << problem.experts() << ',' << problem.flops() << ',' << std::setprecision(12)
                << latency_us << ',' << tflops << ",PASS," << entry.max_abs_error << ','
                << std::quoted(entry.operation->config()) << ',' << std::quoted(environment.gpu) << ','
                << environment.runtime << ',' << environment.driver << ',' << environment.cublas << ','
                << cutlass_version() << ',' << options.device << ',' << options.warmup << ','
                << options.iterations << ',' << options.repeats << ",warm_reused_inputs\n";
        }
        csv.flush();
        if (!csv) {
            throw std::runtime_error("Failed to write " + options.csv);
        }
    }
}

}  // namespace group_gemm
