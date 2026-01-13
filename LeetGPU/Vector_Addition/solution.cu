#include <cuda_runtime.h>

const int NUMTHREAD = 128;

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    const uint gridCol = blockIdx.x;
    const uint threadCol = threadIdx.x;
    if (gridCol * NUMTHREAD + threadCol >= N) return;
    __shared__ float As[NUMTHREAD];
    __shared__ float Bs[NUMTHREAD];
    // 将A, B, C设置到正确位置
    A += gridCol * NUMTHREAD;
    B += gridCol * NUMTHREAD;
    C += gridCol * NUMTHREAD;
    // 加载数据
    As[threadCol] = A[threadCol];
    Bs[threadCol] = B[threadCol];
    // 计算
    float res = As[threadCol] + Bs[threadCol];
    // 写回
    C[threadCol] = res; 
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = NUMTHREAD;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
