// perf.cu
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "warptiling.cuh"

// 辅助函数：检查CUDA错误
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// 性能测试结构体
struct BenchmarkResult {
    float avg_time_ms;
    float min_time_ms;
    float max_time_ms;
    double gflops;
    double bandwidth_gb_s;
};

// 将 benchmarkKernel 改为模板函数
template <const int BM, const int BN, const int BK, const int WM, const int WN, 
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
BenchmarkResult benchmarkKernel(
    dim3 gridDim, dim3 blockDim, int M, int N, int K, float alpha, float *d_A, float *d_B, 
    float beta, float *d_C, int num_warmup = 5, int num_runs = 20) {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < num_warmup; i++) {
        sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    float total_time = 0.0f;
    float min_time = 1e9f;
    float max_time = 0.0f;
    
    for (int i = 0; i < num_runs; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        
        sgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        
        total_time += milliseconds;
        min_time = std::min(min_time, milliseconds);
        max_time = std::max(max_time, milliseconds);
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    BenchmarkResult result;
    result.avg_time_ms = total_time / num_runs;
    result.min_time_ms = min_time;
    result.max_time_ms = max_time;

    // 计算GFLOPS: 2*M*N*K operations
    double flops = 2.0 * M * N * K;
    result.gflops = (flops / result.avg_time_ms) / 1e6;
    
    // 计算有效带宽 (读A, 读B, 写C)
    double bytes = (M * K + K * N + M * N) * sizeof(float);
    result.bandwidth_gb_s = (bytes / result.avg_time_ms) / 1e6;
    
    return result;
}

// 初始化矩阵
void initMatrix(float* mat, int size, float value = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = value;
    }
}

int main() {
    // 矩阵维度
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    
    // 分配host内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // 初始化矩阵
    initMatrix(h_A, M * K, 1.0f);
    initMatrix(h_B, K * N, 1.0f);
    initMatrix(h_C, M * N, 0.0f);

    // 分配device内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // 拷贝数据到device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // 设置kernel参数（编译时常量）
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WNITER = 4;
    constexpr int TM = 8;
    constexpr int TN = 4;
    constexpr int NUM_THREADS = 128;

    // 计算grid和block维度
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));     // 32 x 32
    dim3 blockDim(NUM_THREADS);     // 128

    // GEMM参数
    float alpha = 1.0f;
    float beta = 0.0f;

    // 获取GPU信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    int smCount = prop.multiProcessorCount;
    int coresPerSM = 128;   // 32 x 4
    double clockGHz = prop.clockRate / 1e6;     // 获取时钟频率 (单位为 kHz，转换为 GHz)
    double peakGFLOPS = clockGHz * smCount * coresPerSM * 2;      // FMA = 2 即一个线程每cycle2次Floating Point Operation
    std::cout << "SM Count: " << smCount << std::endl;
    std::cout << "Peak FP32 Performance: " << peakGFLOPS << " GFLOPS" << std::endl;
    std::cout << "Peak Memory Bandwidth: " << (prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2 / 1e6) 
              << " GB/s" << std::endl;
    std::cout << std::endl;

    // 运行benchmark - 使用模板参数
    std::cout << "=== Benchmarking Kernel ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block: " << blockDim.x << std::endl;
    std::cout << std::endl;
    
    BenchmarkResult result = benchmarkKernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>(
        gridDim, blockDim, M, N, K, alpha, d_A, d_B, beta, d_C, 5, 20);
    
    std::cout << "=== Performance Results ===" << std::endl;
    std::cout << "Average time: " << result.avg_time_ms << " ms" << std::endl;
    std::cout << "Min time:     " << result.min_time_ms << " ms" << std::endl;
    std::cout << "Max time:     " << result.max_time_ms << " ms" << std::endl;
    std::cout << "Performance:  " << result.gflops << " GFLOPS" << std::endl;
    std::cout << "Bandwidth:    " << result.bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << std::endl;
    
    // 清理
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    return 0;
}