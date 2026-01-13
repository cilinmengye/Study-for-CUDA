#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

extern "C" void solve(const float* A, const float* B, float* C, int N);

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            std::cerr << "CUDA error: "                     \
                      << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":"          \
                      << __LINE__ << std::endl;             \
            std::exit(EXIT_FAILURE);                        \
        }                                                   \
    } while (0)

int main() {
    const int N = 1 << 20;   // 1M elements
    const int iters = 100;

    std::cout << "N = " << N << ", iters = " << iters << std::endl;

    // ----------------------------
    // Host data
    // ----------------------------
    std::vector<float> hA(N), hB(N), hC(N), hRef(N);

    for (int i = 0; i < N; ++i) {
        hA[i] = float(i);
        hB[i] = float(2 * i);
        hRef[i] = hA[i] + hB[i];
    }

    // ----------------------------
    // Device data
    // ----------------------------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // ----------------------------
    // Warm-up（非常重要）
    // ----------------------------
    for (int i = 0; i < 10; ++i) {
        solve(dA, dB, dC, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------
    // 1. Kernel-only 时间（GPU 时间）
    // ----------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float kernel_ms = 0.0f;

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        solve(dA, dB, dC, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        kernel_ms += ms;
    }

    kernel_ms /= iters;

    // ----------------------------
    // 2. End-to-end 时间（H2D + kernel + D2H）
    // ----------------------------
    float total_ms = 0.0f;

    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        solve(dA, dB, dC, N);

        CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<float, std::milli>(t1 - t0).count();
    }

    total_ms /= iters;

    // ----------------------------
    // Correctness check
    // ----------------------------
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(hC[i] - hRef[i]) > 1e-5f) {
            std::cerr << "Mismatch at " << i
                      << ": got " << hC[i]
                      << ", expected " << hRef[i] << std::endl;
            ok = false;
            break;
        }
    }

    // ----------------------------
    // Report
    // ----------------------------
    double bytes = double(N) * 3 * sizeof(float); // A + B + C
    double kernel_gbps = bytes / (kernel_ms / 1e3) / 1e9;
    double total_gbps  = bytes / (total_ms / 1e3) / 1e9;

    std::cout << "\nCorrectness: " << (ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "Kernel-only time: " << kernel_ms << " ms"
              << "  (" << kernel_gbps << " GB/s)" << std::endl;
    std::cout << "End-to-end time:  " << total_ms << " ms"
              << "  (" << total_gbps << " GB/s)" << std::endl;

    // ----------------------------
    // Cleanup
    // ----------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return ok ? 0 : 1;
}
