#include <cuda_runtime.h>

#define MASK 0xffffffffu
#define CEIL_DIV(M, N) (((M) +  (N - 1)) / (N))
const uint NUM_THREADS = 128;
const uint BM = 16;
const uint BN = 16;
const uint Bd = 16;
const uint WARP_SIZE = 32;

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    // 1. 将内存地址强转为 int 指针，因为 atomicCAS 运行在整数位上
    int* address_as_int = (int*)addr;
    
    // 2. 读取当前内存中的值，并将其解释为 int
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        // 3. 关键点：将位表示还原为 float 进行数学比较
        // 如果当前值已经大于等于我们要写入的值，则退出
        if (__int_as_float(assumed) >= value) {
            break;
        }
        
        // 4. 尝试将新值转换成位表示并写入
        // atomicCAS(地址, 预期旧值, 拟写入新值)
        // 如果在此期间内存被其他线程改了，old 会被更新，循环继续
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void __launch_bounds__(NUM_THREADS) 
softmax_kernel(const float* Q, const float* K, float* qk_output, const int M, 
               const int N, const int d, float* gmem_max, float* gmem_sum) {
    __shared__ float Qs[BM * Bd];
    __shared__ float Ks[Bd * BN];
    __shared__ float smem_qk[BM * BN];
    const uint bRow = blockIdx.y;
    const uint bCol = blockIdx.x;
    const uint tRow = threadIdx.x / BN;
    const uint tCol = threadIdx.x % BN;
    // 将Q,K移动到正确位置
    Q += bRow * BM * d;
    K += bCol * BN * d; // 注意K是转置的

    // 计算矩阵乘法
    float tmp = 0.0;
    for (int i = 0; i < d; i += Bd) {
        // 将Q从GMEM加载到SMEM
        if (bRow * BM + tRow < M && i + tCol < d) Qs[tRow * Bd + tCol] = Q[tRow * d + tCol];
        else Qs[tRow * Bd + tCol] = 0.0;
        // 将K从GMEM加载到SMEM, 注意K是转置的
        if (bCol * BN + tCol < N && i + tRow < d) Ks[tRow * BN + tCol] = K[tCol * d + tRow];
        else Ks[tRow * BN + tCol] = 0.0;
        __syncthreads();

        for (int j = 0; j < Bd; j++) tmp += Qs[tRow * Bd + j] * Ks[j * BN + tCol];
        __syncthreads();

        Q += Bd;
        K += Bd;
    }
    const float sqrtd = sqrtf((float)d);
    float tmpsqrt = tmp / sqrtd;
    smem_qk[tRow * BN + tCol] = tmpsqrt;
    __syncthreads();
    
    // 一个warp处理这件事情：对每一行求出最大值
    const uint warpidx = threadIdx.x / WARP_SIZE;
    if (warpidx == 0) {
        // 每个thread负责8个元素, 求出smem_qk分块中每8个元素中的最大值
        float localmax = -INFINITY;
        const uint wCol = threadIdx.x; // 因为warpidx == 0的现在，现在在此if语句内的threadIdx.x in [0, 31]
        for (int i = 0; i < 8; i++) localmax = fmaxf(localmax, smem_qk[wCol * 8 + i]);

        // 求出smem_qk分块中每行中的最大值
        if (wCol % 2 == 0) localmax = fmaxf(localmax, __shfl_down_sync(MASK, localmax, 1));

        /*
         这里我完全错误了！！！
         因为CUDA 没有任何机制（除了极其复杂的 Cooperative Groups Grid Sync）能让一个 Kernel 里的 Block 等待整个 Grid 的其他 Block 完成更新后再继续执行。
         这里我只能保证threadblock中warp是同步的，但是对于不同的threadblock之间我不能够保证他们是同步的，可能有些threadblock的warp并未准备好localmax
         但是其他threadblock的warp就已经假设gmem_max更新完全了，利用其中的值继续执行, 这完全是错误的
        */

        // 线程wCol % 2 == 0的thread有smem_qk分块中一行的最大值
        // 如下操作写入每分块每行最大值到gmem_max中
        if (wCol % 2 == 0) atomicMaxFloat(&gmem_max[bRow * BM + wCol / 2], localmax);

        // 接下来求sum
         // 每个thread负责8个元素, 求出smem_qk分块中每8个元素中的sum
        float localsum = 0.0;
        float rowmax = gmem_max[bRow * BM + wCol / 2];
        for (int i = 0; i < 8; i++) localsum += __expf(smem_qk[wCol * 8 + i] - rowmax);

        // 求出smem_qk分块中每行中的sum
        if (wCol % 2 == 0) localsum += __shfl_down_sync(MASK, localsum, 1);

        // 线程wCol % 2 == 0的thread有smem_qk分块中一行的sum
        // 如下操作写入每分块每行sum到gmem_max中
        if (wCol % 2 == 0) atomicAdd(&gmem_sum[bRow * BM + wCol / 2], localsum);
    }

    // 然后求softmax
    tmp = __expf(tmpsqrt - gmem_max[bRow * BM + tRow]) / gmem_sum[bRow * BM + tRow];
    // 将smem_qk写回qk_output
    const uint row = bRow * BM + tRow;
    const uint col = bCol * BN + tCol;
    if (row < M && col < M) qk_output[row * N + col] = tmp;
}

__global__ void __launch_bounds__(NUM_THREADS) 
mutal_kernel(const float* Q, const float* K, float* qk_output, const int M, 
               const int N, const int d) {
    __shared__ float Qs[BM * Bd];
    __shared__ float Ks[Bd * BN];
    __shared__ float smem_qk[BM * BN];
    const uint bRow = blockIdx.y;
    const uint bCol = blockIdx.x;
    const uint tRow = threadIdx.x / BN;
    const uint tCol = threadIdx.x % BN;
    // 将Q,K移动到正确位置
    Q += bRow * BM * d;
    K += bCol * BN * d; // 注意K是转置的

    // 计算矩阵乘法
    float tmp = 0.0;
    for (int i = 0; i < d; i += Bd) {
        // 将Q从GMEM加载到SMEM
        if (bRow * BM + tRow < M && i + tCol < d) Qs[tRow * Bd + tCol] = Q[tRow * d + tCol];
        else Qs[tRow * Bd + tCol] = 0.0;
        // 将K从GMEM加载到SMEM, 注意K是转置的
        if (bCol * BN + tCol < N && i + tRow < d) Ks[tRow * BN + tCol] = K[tCol * d + tRow];
        else Ks[tRow * BN + tCol] = 0.0;
        __syncthreads();

        for (int j = 0; j < Bd; j++) tmp += Qs[tRow * Bd + j] * Ks[j * BN + tCol];
        __syncthreads();

        Q += Bd;
        K += Bd;
    }
    const float sqrtd = sqrtf((float)d);
    smem_qk[tRow * BN + tCol] = tmp / sqrtd;
    __syncthreads();

    // 将smem_qk写回qk_output
    const uint row = bRow * BM + tRow;
    const uint col = bCol * BN + tCol;
    if (row < M && col < M) qk_output[row * N + col] = smem_qk[tRow * BN + tCol]; 
}



// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float* gmem_max = nullptr;
    float* gmem_sum = nullptr;
    float* qk_output = nullptr;
    cudaMalloc(&gmem_max, sizeof(M));
    cudaMalloc(&gmem_sum, sizeof(M));
    cudaMalloc(&qk_output, sizeof(M * N));
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(NUM_THREADS);

    softmax_kernel<<<gridDim, blockDim>>>(Q, K, qk_output, M, N, d, gmem_max, gmem_sum);

    cudaDeviceSynchronize();
}

extern "C" void debug_mutal_solve(const float* Q, const float* K, const float* V, float* qk_output, int M, int N, int d) {
    float* gmem_max = nullptr;
    float* gmem_sum = nullptr;
    cudaMalloc(&gmem_max, sizeof(M));
    cudaMalloc(&gmem_sum, sizeof(M));
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(NUM_THREADS);

    // softmax_kernel<<<gridDim, blockDim>>>(Q, K, qk_output, M, N, d, gmem_max, gmem_sum);
    mutal_kernel<<<gridDim, blockDim>>>(Q, K, qk_output, M, N, d);
    cudaDeviceSynchronize();
}

extern "C" void debug_softmax_solve(const float* Q, const float* K, const float* V, float* qk_output, int M, int N, int d) {
    float* gmem_max = nullptr;
    float* gmem_sum = nullptr;
    cudaMalloc(&gmem_max, sizeof(M));
    cudaMalloc(&gmem_sum, sizeof(M));
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(NUM_THREADS);

    // softmax_kernel<<<gridDim, blockDim>>>(Q, K, qk_output, M, N, d, gmem_max, gmem_sum);
    softmax_kernel<<<gridDim, blockDim>>>(Q, K, qk_output, M, N, d, gmem_max, gmem_sum);
    cudaDeviceSynchronize();
}


/*
softmax_kernel完全错误，原因在其中的注释也有写，下面再写下：

你的逻辑：

Block (0,0) 计算出子块结果，更新全局 gmem_max 和 gmem_sum。

Block (0,0) 立刻读取 gmem_sum 做除法： tmp = ... / gmem_sum[...]。

写回 qk_output。

现实情况： 当 Block (0,0) 执行第 2 步（做除法）的时候，负责同一行的 Block (1,0), Block (2,0)... 可能连启动都还没启动！ 或者正在运行中，还没有来得及把它们的数值 atomicAdd 到 gmem_sum 里。

后果： Block (0,0) 做除法时使用的 gmem_sum 是不完整的（Partial Sum）。它只包含了它自己那一部分，或者运气好包含了前面跑完的几个 Block 的部分。 CUDA 没有任何机制（除了极其复杂的 Cooperative Groups Grid Sync）能让一个 Kernel 里的 Block 等待整个 Grid 的其他 Block 完成更新后再继续执行。

结论： 你不能在同一个 Kernel 里先计算 Global Sum，然后立刻在同一行代码里读取这个 Global Sum 做归一化。这必须拆分成两个 Kernel，或者改变算法结构（让一个 Block 处理整行）。

接下来，除非使用flash attention，否则我还老实按照分步写法来吧：
1. 首先求得QK^T / sqrt(d) 矩阵
2. 然后求得每行Max和Sum
3. 然后求softmax
4. 乘以V
*/