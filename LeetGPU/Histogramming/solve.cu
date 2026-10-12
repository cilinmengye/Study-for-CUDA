#include <cuda_runtime.h>

#define C_DIV(M,N) (((M)+(N)-1)/(N))

template<int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
histogram_kernel(
    const int* input,
    int* histogram,
    int N,
    int num_bins
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offs = bid * NUM_THREADS + tid;
    int stride = gridDim.x * NUM_THREADS;

    extern __shared__ int shared_histogram[];

    // 清零 shared_histogram
    for (int i = tid; i < num_bins; i += NUM_THREADS) {
        shared_histogram[i] = 0;
    }
    __syncthreads();


    for (int i = offs; i < N; i += stride) {
        int num = input[i];
        // 我们需要让线程原子地在 shared_histogram 相加
        atomicAdd(&shared_histogram[num], 1);
    }
    __syncthreads();

    // 然后我们需要让 shared_histogram 原子地在 histogram 上相加.
    for (int i = tid; i < num_bins; i += NUM_THREADS) {
        atomicAdd(&histogram[i], shared_histogram[i]);
    }
}


// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    constexpr int NUM_THREADS = 128;
    constexpr int MAX_BLOCKS = 4096;
    size_t shared_bytes = static_cast<size_t>(num_bins) * sizeof(int);

    // 我们不开过多的 block 和 thread
    // 开过多 block, 初始化过程时每个 block 都需要初始化 kernel 内部操作, 耗时
    // 开过多 thread, 大部分 thread 会不在运行.
    int num_blocks = C_DIV(N, NUM_THREADS);
    if (num_blocks > MAX_BLOCKS) {
        num_blocks = MAX_BLOCKS;
    }

    dim3 grid = (num_blocks);
    dim3 block = (NUM_THREADS);
    histogram_kernel<NUM_THREADS><<<grid, block, shared_bytes>>>(
        input,
        histogram,
        N,
        num_bins
    );
}
