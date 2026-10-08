#include <cuda_runtime.h>

#define C_DIV(M, N) (((M) + (N) - 1) / (N))

template<int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
reduction_sum_kernel(
    const float* input,
    float* output,
    int N
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int b_offs = bid * NUM_THREADS;

    __shared__ float tile_arr[NUM_THREADS];

    // 一个 Block 中的全部 Thread 一起先把数据从内存中放到 shared memory 中
    // 因为一个 Block 负责 NUM_THREADS 个元素, 同时我们也有 NUM_THREADS 个 thread
    // 所以一个 thread 负责相应元素
    const float* input_block_ptr = input + b_offs;
    const float* input_thread_ptr = input_block_ptr + tid;
    bool mask = (b_offs + tid) < N;
    if (mask) {
        tile_arr[tid] = input_thread_ptr[0];
    } else {
        tile_arr[tid] = 0.0f;
    }

    // 等线程全部读取完成
    __syncthreads();

    // 通过分治进行并行相加
    int offs = NUM_THREADS / 2;
    while (offs > 0) {
        if (tid < offs) {
            int other_tid = tid + offs;
            
            // 我们已经把越界的 tile_arr 给设置为 0 了, 所以这里不用判断也行
            tile_arr[tid] += tile_arr[other_tid];
        }
        offs /= 2;
        // 这里因为可能是不同 warp 中线程，所以我们也需要在这里等一下
        __syncthreads();
    }

    // 然后一个 block 线程全部算完了，需要合并其他 block 的数据
    if (tid == 0) {
        // 原子加法
        atomicAdd(output, tile_arr[0]);
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));
    constexpr int NUM_THREADS = 128;
    dim3 block = (NUM_THREADS);
    dim3 grid  = (C_DIV(N, NUM_THREADS));
    
    reduction_sum_kernel<NUM_THREADS>
        <<<grid, block>>>(input, output, N);

    cudaDeviceSynchronize();
}
