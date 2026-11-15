/*
 * benchmark_runner.h
 * 场景运行入口与运行环境描述；供 main() 调用边界检查和正式测量。
 */
#pragma once

#include "benchmark_options.h"
#include "testbed.h"

#include <iosfwd>

namespace group_gemm {

/* 记录 GPU 与库版本，供控制台和 CSV 使用。 */
struct Environment {
    std::string gpu;
    int runtime;
    int driver;
    int cublas;
};


void check_edges(const Options& options, DType dtype, cudaStream_t stream);
void run_case(
    const CaseSpec& spec, DType dtype, const Options& options,
    cudaStream_t stream, const Environment& environment, std::ofstream& csv
);

}  // namespace group_gemm
