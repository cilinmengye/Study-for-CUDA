#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "warptiling_threadtiling.cuh"

// 验证函数
bool verifyResult(float* C, int M, int N, int K, float tolerance = 1e-3) {
    // 简单验证：如果A和B都是1，C应该都是K
    float expected = (float)K;
    
    int errors = 0;
    for (int i = 0; i < M * N && errors < 10; i++) {
        if (fabs(C[i] - expected) > tolerance) {
            std::cout << "Error at [" << i/N << "][" << i%N << "]: "
                      << "expected " << expected << ", got " << C[i] << std::endl;
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "✓ Verification passed!" << std::endl;
        return true;
    } else {
        std::cout << "✗ Found " << errors << " errors" << std::endl;
        return false;
    }
}

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

// 初始化矩阵
void initMatrix(float* mat, int size, float value = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = value;
    }
}

int main() {
    // printf("%d\n", ISUSE);
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

    // 设置kernel参数
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 64;
    const int TM = 16
    const int TN = 8;
    const int TK = 2;
    const int NUM_THREADS = 128;

    // 计算grid和block维度
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));     // 32 x 32
    dim3 blockDim(NUM_THREADS);     // 128

    // GEMM参数
    float alpha = 1.0f;
    float beta = 0.0f;

    // 启动kernel
    sgemmWarptiling<BM, BN, BK, WM, WN, TM, TN, TK, NUM_THREADS>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    
    // 检查kernel启动错误
    CHECK_CUDA(cudaGetLastError());
    
    // 等待kernel完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 拷贝结果回host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // 验证
    verifyResult(h_C, M, N, K);
    
    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    std::cout << "Done!" << std::endl;
    return 0;
}