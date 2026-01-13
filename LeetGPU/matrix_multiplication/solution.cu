#include <cuda_runtime.h>

#define CIEL_DIV(M, N) (((M) + (N) - 1) / (N))
const uint BM = 16;
const uint BK = 16;
const uint BN = 16;
const uint NUM_THREADS = 256;

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    
    __shared__ float As[BM][BN];
    __shared__ float Bs[BN][BK];
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint tRow = threadIdx.x / BN;
    const uint tCol = threadIdx.x % BN;

    // Move A, B, C to the correct position
    A += cRow * BM * N;
    B += cCol * BK;
    C += cRow * BM * K + cCol * BK;

    float tmp = 0.0;    // 每个线程负责一个元素
    for (int n = 0; n < N; n += BN) {
        if (cRow * BM + tRow < M && n + tCol < N) As[tRow][tCol] = A[tRow * N + tCol]; // 将A从GMEM加载到SMEM的As
        else As[tRow][tCol] = 0.0;

        if (n + tRow < N && cCol * BK + tCol < K) Bs[tRow][tCol] = B[tRow * K + tCol]; // 将B从GMEM加载到SMEM的Bs
        else Bs[tRow][tCol] = 0.0;

        // blockthread线程同步, 下面的thread需要用到其他thread加载的数据
        __syncthreads();

        // 计算
        for (int ns = 0; ns < BN; ns++) tmp += As[tRow][ns] * Bs[ns][tCol];
        // 防止As和Bs正在被某个线程使用，但是某个线程提前再去从GMEM中加载数据到As和Bs中造成写后读冲突
        __syncthreads();
    
        A += BN;
        B += BN * K;
    }

    // 写回
    if (cRow * BM + tRow < M && cCol * BK + tCol < K) C[tRow * K + tCol] = tmp;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 gridDim(CIEL_DIV(K, BK), CIEL_DIV(M, BM));
    dim3 blockDim(NUM_THREADS);

    matrix_multiplication_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
