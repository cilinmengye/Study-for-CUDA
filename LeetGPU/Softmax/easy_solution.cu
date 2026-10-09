#include <cuda_runtime.h>
#include <cfloat>

// 核心思想是让每个 Block 都会遍历整个输入。例如，全部 Block 的线程 0 都读取下标 0、256、512……。
// 每个 Block 都有 input 的最大值, 不需要再用额外 block 之间的同步方式。
template<int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
softmax_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    __shared__ float shared_max[NUM_THREADS];

    // 求出每个 thread 负责数据的最大值
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += NUM_THREADS) {
        local_max = fmax(local_max, input[i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();
    // 现在相当于整个 input 的段落内的最大值都在 shared_max 里面了.
    // 然后需要在 block 的 thread 内部求最大值
    // 分治
    for (int i = NUM_THREADS / 2; i > 0; i /= 2) {
        if (tid < i) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + i]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    // 对于每个 block 而言, input 的最大值现在在 shared_max[0] 上.
    // 所以此时相当于每个 block 都有 input 的最大值.
    // 然后我们可以开始求 e^x.
    __shared__ float shared_sum[NUM_THREADS];
    float localsum = 0.0f;
    for (int i = tid; i < N; i += NUM_THREADS) {
        localsum += __expf(input[i] - max_val);
    }
    shared_sum[tid] = localsum;
    __syncthreads();
    
    for (int i = NUM_THREADS / 2; i > 0; i /= 2) {
        if (tid < i) {
            shared_sum[tid] += shared_sum[tid + i];
        }
        __syncthreads();
    }
    float sum_exp = shared_sum[0];
    __syncthreads();

    for(int i = tid; i < N; i += NUM_THREADS) {
        output[i] = __expf(input[i] - max_val) / sum_exp;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<threadsPerBlock>
        <<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
