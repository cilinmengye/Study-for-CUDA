#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

extern "C" void debug_reduce_max_kernel(const float* input, float* output, int N);
extern "C" void debug_reduce_sum_kernel(const float* input, float* output, int N);
extern "C" void solve(const float* input, float* output, int N);

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n",
                msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    constexpr int N = 3;
    // -------- host data --------
    std::vector<float> h_input(N);
    h_input[0] = -1.0; h_input[1] = -2.0; h_input[2] = -3.0;
    std::vector<float> h_output(N);
    // -------- device data --------
    float *d_input = nullptr, *d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, N * sizeof(float)), "cudaMalloc d_input");
    check_cuda(cudaMalloc(&d_output, N * sizeof(float)), "cudaMalloc d_output");
    check_cuda(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    printf("start debug_reduce_max_kernel\n");
    debug_reduce_max_kernel(d_input, d_output, N);
    printf("start debug_reduce_sum_kernel\n");
    debug_reduce_sum_kernel(d_input, d_output, N);
    printf("start solve");
    solve(d_input, d_output, N);

    check_cuda(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    printf("output:");
    for (int i = 0; i < N; i++) printf("%f ", h_output[i]);
    printf("\n");
}