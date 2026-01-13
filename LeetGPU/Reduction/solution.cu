#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
const uint NUM_THREADS = 128;
const uint TN = 8;
const uint NUM_PER_BLOCK = NUM_THREADS * TN;    // blockthread处理的元素数量

__global__ void reduction(const float* input, float* output, int N){
    __shared__ float Ns[NUM_THREADS];
    const uint idx = blockIdx.x * NUM_PER_BLOCK + threadIdx.x * TN;
    const uint tidx = threadIdx.x;
    
    // 每个线程处理TN个元素
    float tmp = 0.0;
    for (int i = 0; i < TN; i++) {
        if (idx + i < N) {  // 防止input out of bound
            tmp += input[idx + i];        
        }
    }
    Ns[tidx] = tmp;
    __syncthreads();

    // blockthread之内的归约
    for (int i = NUM_THREADS / 2; i > 0; i >>= 1) {     // "对折"处理
        if (tidx < i) {
            Ns[tidx] += Ns[tidx + i];
        }
        __syncthreads();
    }

    // 每个blockthread负责处理blockthread的Ns[0]
    if (tidx == 0) atomicAdd(output, Ns[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 blockDim(NUM_THREADS);
    dim3 gridDim(CEIL_DIV(N, NUM_PER_BLOCK));
    reduction<<<gridDim, blockDim>>>(input, output, N);
}
