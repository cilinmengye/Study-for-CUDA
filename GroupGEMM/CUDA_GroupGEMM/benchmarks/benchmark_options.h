/*
 * benchmark_options.h
 * 命令行选项与解析入口；保留原有 Options 字段、默认值和函数名。
 */
#pragma once

#include <string>

namespace group_gemm {

// ===========================
// 命令行与结果路径
// ===========================

/* iterations 是每个计时批次的完整 Group GEMM 次数，repeats 是独立采样轮数。 */
struct Options {
    std::string dtype = "all";
    std::string scenario = "all";
    std::string provider = "all";
    std::string timing = "both";
    std::string csv = "results.csv";
    int device = 0;
    int warmup = 20;
    int iterations = 100;
    int repeats = 7;
    bool check_only = false;
    bool help = false;
};


/* argv: [Host] 标准进程参数；结果路径规则由 parse_options() 统一处理。 */
Options parse_options(int argc, char** argv);
void print_help();
bool wants(const Options& options, const std::string& provider);

}  // namespace group_gemm
