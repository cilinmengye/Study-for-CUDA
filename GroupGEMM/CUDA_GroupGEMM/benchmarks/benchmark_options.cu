/*
 * benchmark_options.cu
 * 命令行解析、结果路径和帮助信息；原函数体从 benchmarks.cu 原样迁移。
 */
#include "benchmark_options.h"

#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <stdexcept>

namespace group_gemm {

/* argv: [Host] 标准进程参数；只暴露学习和复现实验需要的选项。 */
Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            // 帮助选项不消耗后续参数，也不需要初始化 GPU。
            options.help = true;
            continue;
        }
        if (key == "--check-only") {
            // 正确性开关没有参数值。
            options.check_only = true;
            continue;
        }
        if (i + 1 == argc) {
            throw std::runtime_error("Missing value for " + key);
        }

        const std::string value = argv[++i];
        const auto integer = [&]() {
            size_t consumed = 0;
            const int number = std::stoi(value, &consumed);
            if (consumed != value.size()) {
                throw std::runtime_error("Invalid integer for " + key);
            }
            return number;
        };
        if (key == "--dtype") options.dtype = value;
        else if (key == "--scenario") options.scenario = value;
        else if (key == "--provider") options.provider = value;
        else if (key == "--timing") options.timing = value;
        else if (key == "--csv") options.csv = value;
        else if (key == "--device") options.device = integer();
        else if (key == "--warmup") options.warmup = integer();
        else if (key == "--iterations") options.iterations = integer();
        else if (key == "--repeats") options.repeats = integer();
        else throw std::runtime_error("Unknown option " + key);
    }

    const auto one_of = [](const std::string& value, std::initializer_list<const char*> allowed) {
        return std::any_of(allowed.begin(), allowed.end(), [&](const char* item) { return value == item; });
    };
    if (!one_of(options.dtype, {"all", "fp16", "bf16"}) ||
        !one_of(options.scenario, {"all", "prefill", "decode"}) ||
        !one_of(options.provider, {"all", "cublas", "cutlass", "custom"}) ||
        !one_of(options.timing, {"both", "normal", "graph"}) ||
        options.device < 0 || options.warmup < 1 || options.iterations < 1 || options.repeats < 1) {
        throw std::runtime_error("Invalid options; use --help");
    }
#ifndef ENABLE_CUSTOM
    if (options.provider == "custom") {
        throw std::runtime_error("Custom backend is not built; implement custom_gemm.h and enable ENABLE_CUSTOM");
    }
#endif
    // NOTE: CMake 记录项目 results/ 的绝对位置，因此从项目根目录或 build/ 启动
    //       都写入同一目录。相对 --csv 路径以 results/ 为基准，绝对路径按用户指定。
    //       此处只计算路径；--help / --check-only 不创建目录或文件。
    std::filesystem::path output_path(options.csv);
    if (output_path.is_relative()) {
        output_path = std::filesystem::path(GROUP_GEMM_RESULTS_DIR) / output_path;
    }
    options.csv = output_path.lexically_normal().string();
    return options;
}


void print_help() {
    std::cout
        << "group_gemm_bench [options]\n"
        << "  --dtype all|fp16|bf16       --scenario all|prefill|decode\n"
        << "  --provider all|cublas|cutlass|custom\n"
        << "  --timing both|normal|graph --device 0\n"
        << "  --warmup 20 --iterations 100 --repeats 7\n"
        << "  --csv results.csv         --check-only\n"
        << "Relative --csv paths are rooted at: " << GROUP_GEMM_RESULTS_DIR << '\n'
        << "Defaults: both dtypes, both scenarios, all compiled providers, both timing modes.\n"
        << "Graph tuning always precedes timed CUTLASS results; --check-only does not tune.\n";
}


bool wants(const Options& options, const std::string& provider) {
    return options.provider == "all" || options.provider == provider;
}

}  // namespace group_gemm
