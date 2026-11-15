/*
 * benchmarks.cu
 * 统一入口：先验证，再筛选 CUTLASS 配置，最后分别测普通调用与 CUDA Graph。
 * 延迟单位 us，吞吐量按有效 FLOPs / 时间计算；不计入 padding 的额外工作。
 * 阅读主线：main -> check_edges -> run_case -> tune_cutlass / BatchTimer。
 * 一个 operation.run() 表示整组专家 GEMM，不是单个专家或单个 threadblock。
 */

/*
 * NOTE: 本文件保留主流程；选项在 benchmark_options，计时在 benchmark_timing，
 *       场景与调优在 benchmark_runner，custom 适配器在 custom_operation。
 *       原阅读说明中的 tune_cutlass 对应现在的 tune_candidates，函数名保持现状。
 */
#include "benchmark_options.h"
#include "benchmark_runner.h"
#include "benchmark_timing.h"

#include <filesystem>
#include <fstream>
#include <iostream>


/* argv: [Host] 命令行参数；主程序只拥有测试资源，不实现任何 GEMM kernel。 */
int main(int argc, char** argv) {
    using namespace group_gemm;
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            // 无 GPU 的环境也能查看命令行说明。
            print_help();
            return 0;
        }

        CUDA_CHECK(cudaSetDevice(options.device));
        // 先选择设备再创建 stream、handle 和显存，避免不同设备的资源被混用。
        Stream stream;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
        Environment environment{properties.name, 0, 0, 0};
        CUDA_CHECK(cudaRuntimeGetVersion(&environment.runtime));
        CUDA_CHECK(cudaDriverGetVersion(&environment.driver));
        {
            BlasHandle handle(stream.value);
            CUBLAS_CHECK(cublasGetVersion(handle.get(), &environment.cublas));
        }
        std::cout << "GPU " << options.device << ": " << environment.gpu
                  << " (SM " << properties.major << '.' << properties.minor << ")\n"
                  << "CUDA runtime=" << environment.runtime << " driver=" << environment.driver
                  << " cuBLAS=" << environment.cublas << " CUTLASS=" << cutlass_version() << '\n'
                  << "构建时间: " << __DATE__ << ' ' << __TIME__ << '\n'
                  << "FP16/BF16 输入输出，FP32 累加；预热后重复使用输入，未清空 L2。\n";
#ifndef ENABLE_CUSTOM
        std::cout << "custom: SKIP（用户 kernel 尚未接入）\n";
#endif

        std::ofstream csv;
        if (!options.check_only) {
            // 只为性能任务创建输出，避免正确性命令覆盖之前的结果。
            // NOTE: results/ 被 Git 忽略，新 checkout 中可能不存在；同时允许
            //       --csv experiment/run.csv 自动建立 results/experiment/。
            std::filesystem::create_directories(std::filesystem::path(options.csv).parent_path());
            csv.open(options.csv);
            if (!csv) {
                throw std::runtime_error("Cannot open " + options.csv);
            }
            csv << "scenario,layout,dtype,provider,timing,N,K,experts,flops,latency_us,tflops,correctness,"
                << "max_abs_error,config,gpu,cuda_runtime,cuda_driver,cublas,cutlass,device,warmup,iterations,repeats,cache_policy\n";
        }

        for (DType dtype : {DType::FP16, DType::BF16}) {
            if (options.dtype != "all" && options.dtype != dtype_name(dtype)) {
                continue;
            }

            // 先用小尺寸独立 CPU 参考验证边界，再运行两个真实尺寸场景。
            // --check-only 仍会覆盖主场景的所有候选，但 run_case 会在计时前返回。
            check_edges(options, dtype, stream.value);
            for (const auto& spec : primary_cases()) {
                if (options.scenario == "all" || options.scenario == spec.name) {
                    run_case(spec, dtype, options, stream.value, environment, csv);
                }
            }
        }
        std::cout << (options.check_only ? "\n所有选定的正确性检查通过。\n" : "\n测试完成，CSV: " + options.csv + "\n");
        return 0;
    } catch (const std::exception& error) {
        // 任何 CUDA/数值/guard 故障均导致非零退出，不发布该失败操作的性能数据。
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
