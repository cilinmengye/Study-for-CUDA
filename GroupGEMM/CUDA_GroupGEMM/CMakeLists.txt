cmake_minimum_required(VERSION 3.24)
project(cuda_group_gemm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CUTLASS_DIR "" CACHE PATH "CUTLASS root containing include/cutlass")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/gemm/device/gemm_grouped.h")
    message(FATAL_ERROR "Set -DCUTLASS_DIR=/path/to/cutlass; see README.md")
endif()

add_executable(group_gemm_bench
    benchmarks/benchmarks.cu
    benchmarks/benchmark_options.cu
    benchmarks/benchmark_timing.cu
    benchmarks/benchmark_runner.cu
    tests/test_correctness.cu
    src/cublas_grouped_gemm.cu
    src/cutlass_grouped_gemm.cu
)
# include/ 是公共接口，tests/ 中的 testbed.h 由测试与 benchmark 共用。
# 使用 include 搜索路径后，源文件不需要写容易随移动失效的 ../ 相对路径。
target_include_directories(group_gemm_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)
target_include_directories(group_gemm_bench SYSTEM PRIVATE "${CUTLASS_DIR}/include")
target_compile_definitions(group_gemm_bench PRIVATE
    GROUP_GEMM_RESULTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/results"
)
target_link_libraries(group_gemm_bench PRIVATE CUDA::cudart CUDA::cublas ${CMAKE_DL_LIBS})
set_target_properties(group_gemm_bench PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 89
)
target_compile_options(group_gemm_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;--expt-relaxed-constexpr>
)

# 用户自行提供两个 launch 函数；默认构建不需要任何自定义 kernel。
option(ENABLE_CUSTOM "Build the custom contiguous/masked CUDA kernels" OFF)
if(ENABLE_CUSTOM)
    # 保留当前实现的文件名，两个目标使用同一份 kernel 源码。
    # tilegemm/tilegemm_v0.cu 和 tilegemm_v1.cu 是被包含的设备模板定义，不单独编译。
    set(CUSTOM_GEMM_SOURCE src/cutsom_gemm.cu)
    target_sources(group_gemm_bench PRIVATE ${CUSTOM_GEMM_SOURCE})
    target_compile_definitions(group_gemm_bench PRIVATE ENABLE_CUSTOM=1)

    add_executable(custom_gemm_test
        tests/test_custom_gemm.cu
        tests/test_correctness.cu
        ${CUSTOM_GEMM_SOURCE}
    )
    target_include_directories(custom_gemm_test PRIVATE include tests)
    target_link_libraries(custom_gemm_test PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(custom_gemm_test PROPERTIES
        CXX_STANDARD 17
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES 89
    )
    target_compile_options(custom_gemm_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>
    )
endif()

enable_testing()
add_test(NAME grouped_gemm_correctness COMMAND group_gemm_bench --check-only)
if(ENABLE_CUSTOM)
    add_test(NAME custom_gemm_correctness COMMAND custom_gemm_test)
endif()
