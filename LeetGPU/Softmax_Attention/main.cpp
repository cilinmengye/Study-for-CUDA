// main.cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while (0)

// print helper
void print_matrix(const std::vector<float>& A, int rows, int cols, const char* name) {
    std::cout << name << " (" << rows << " x " << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << A[i*cols + j];
            if (j + 1 < cols) std::cout << ", ";
        }
        std::cout << "\n";
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

int main() {
    // small reproducible test
    // const int M = 2;
    // const int N = 3;
    // const int d = 4;

    // // host data (row-major)
    // std::vector<float> Q = {1.0, 0,0, 0.0, 0.0, 
    //                         0.0, 1.0, 0.0, 0.0};     // M x d
    // std::vector<float> K = {1.0, 0,0, 0.0, 0.0,
    //                         0.0, 1.0, 0.0, 0.0,
    //                         0.0, 0.0, 1.0, 0.0}; // N x d
    // std::vector<float> V = {1, 2, 3, 4,
    //                         5, 6, 7, 8,
    //                         9, 10, 11, 12}; // N x d
    const int M = 4;
    const int N = 4;
    const int d = 3;
    std::vector<float> Q = {
        -1.0f,  2.0f, -3.0f,
        4.0f, -5.0f,  6.0f,
        -7.0f,  8.0f, -9.0f,
        10.0f, -11.0f, 12.0f
    }; // M=4 x d=3

    std::vector<float> K = {
        2.0f, -1.0f,  3.0f,
        -4.0f,  5.0f, -6.0f,
        7.0f, -8.0f,  9.0f,
        -10.0f, 11.0f, -12.0f
    }; // N=4 x d=3

    std::vector<float> V = {
        1.0f,  0.5f, -0.5f,
        -1.0f,  2.0f,  3.0f,
        4.0f, -2.0f,  1.0f,
        0.0f,  1.0f, -1.0f
    }; // N=4 x d_v=3

    // device buffers
    float *Q_d=nullptr, *K_d=nullptr, *V_d=nullptr;
    float *qk_d=nullptr, *out_d=nullptr;

    CHECK_CUDA(cudaMalloc(&Q_d, sizeof(float)*M*d));
    CHECK_CUDA(cudaMalloc(&K_d, sizeof(float)*N*d));
    CHECK_CUDA(cudaMalloc(&V_d, sizeof(float)*N*d));
    // qk_output is M x N, output is M x d
    CHECK_CUDA(cudaMalloc(&qk_d, sizeof(float)*M*N));
    CHECK_CUDA(cudaMalloc(&out_d, sizeof(float)*M*d));

    // copy inputs
    CHECK_CUDA(cudaMemcpy(Q_d, Q.data(), sizeof(float)*M*d, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K_d, K.data(), sizeof(float)*N*d, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V_d, V.data(), sizeof(float)*N*d, cudaMemcpyHostToDevice));


    solve(Q_d, K_d, V_d, out_d, M, N, d);
    std::vector<float> out_h(M*d);
    CHECK_CUDA(cudaMemcpy(out_h.data(), out_d, sizeof(float)*M*d, cudaMemcpyDeviceToHost));
    print_matrix(out_h, M, d, "OUTPUT");
    // // call your solve (device-side implementation must write into qk_output and output)
    // debug_mutal_solve(Q_d, K_d, V_d, qk_d, M, N, d);

    // // synchronize and check errors
    // CHECK_CUDA(cudaDeviceSynchronize());
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "Kernel launch error: "
    //           << cudaGetErrorString(err) << std::endl;
    //     return 1;
    // }

    // // copy back results
    // std::vector<float> qk_h(M*N);
    // CHECK_CUDA(cudaMemcpy(qk_h.data(), qk_d, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    // // print
    // print_matrix(qk_h, M, N, "debug_mutal_solve qk_output (device)");
    
    // // ==============================测试debug_softamx_solve=============================================

    // debug_softmax_solve(Q_d, K_d, V_d, qk_d, M, N, d);
    // // synchronize and check errors
    // CHECK_CUDA(cudaDeviceSynchronize());
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "Kernel launch error: "
    //           << cudaGetErrorString(err) << std::endl;
    //     return 1;
    // }
    // CHECK_CUDA(cudaMemcpy(qk_h.data(), qk_d, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    // // print
    // print_matrix(qk_h, M, N, "debug_softamx_solve qk_output (device)");


    // std::vector<float> out_h(M*d);
    // CHECK_CUDA(cudaMemcpy(out_h.data(), out_d, sizeof(float)*M*d, cudaMemcpyDeviceToHost));
    // print_matrix(out_h, M, d, "output (device)");
    // debug_softamx_solve(Q_d, K_d, V_d, out_d, M, N, d);
    // cleanup
    CHECK_CUDA(cudaFree(Q_d));
    CHECK_CUDA(cudaFree(K_d));
    CHECK_CUDA(cudaFree(V_d));
    CHECK_CUDA(cudaFree(qk_d));
    CHECK_CUDA(cudaFree(out_d));
    return 0;
}