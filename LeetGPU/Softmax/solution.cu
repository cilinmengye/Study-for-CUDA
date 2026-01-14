#include <cuda_runtime.h>
#include <cstdio>

#define FULL_MASK 0xffffffffu
#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))
const uint NUM_THREADS = 1024;
const uint WARP_SIZE = 32;
const uint NUM_WARP = NUM_THREADS / WARP_SIZE;  // 这里有个隐藏条件 NUM_WARP <= WARP_SIZE

/*
浮点数的痛点：float 类型的 atomicMax
这是 CUDA 开发中最容易踩坑的地方。

情况 A：现代显卡 (Compute Capability 8.0+, 如 A100, RTX 30/40系列)
NVIDIA 终于在硬件层面原生支持了 float 类型的 atomicMax。 你不需要做任何特殊处理，直接用：

C++

__global__ void my_kernel(float* addr) {
    float val = 3.14f;
    atomicMax(addr, val); // 编译器会自动生成原生指令 reduce.max.f32
}
情况 B：老旧显卡 (Pascal, Volta, Turing 等，如 GTX 1080, RTX 2080)
这些架构不支持浮点数的原子最大值操作。如果你直接写 atomicMax(float_ptr, val)，编译器会报错。

为了解决这个问题，开发者们发明了两种方法：

方法 1：类型欺骗法 (Type Punning / Bit Casting) 这就是你代码里用到的 __float_as_int 写法。

C++

atomicMax((int*)address, __float_as_int(val));
原理：利用 IEEE 754 浮点数的一个特性——正浮点数的二进制位表示（当做整数看时）的大小顺序，与浮点数本身的数值大小顺序是一致的。

例如：1.0 < 2.0，且 int_rep(1.0) < int_rep(2.0)。

致命缺陷：只对正数有效！

如果你试图对负数用这个方法，结果是反的。因为负数在整数补码表示下，绝对值越大的负数（比如 -5.0），其整数位表示反而看起来更大（或者比较逻辑混乱），导致选出了“数值最小”的数。

结论：除了全正数的 ReLU 输出等场景，严禁使用此方法。

方法 2：通用 CAS 循环法 (Atomic Compare-And-Swap) 这是官方推荐的、在老显卡上处理 float 的标准做法。它利用 atomicCAS 构建一个自旋更新逻辑。
优点：逻辑绝对正确，支持正负数、Infinity 等。

缺点：比原生指令慢，因为需要循环尝试。
*/
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

__device__ float warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 16));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 8));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 4));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 2));
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, 1));
    return v;
}

// 获得数组中最大值，保存到全局内存gmax
__global__ void reduce_max_kernel(const float* input, float* gmax, const int N) {
    const uint bidx = blockIdx.x;
    const uint tidx = threadIdx.x;
    const uint wRow = tidx / WARP_SIZE;
    const uint wCol = tidx % WARP_SIZE;
    const uint arridx = bidx * NUM_THREADS + tidx;

    __shared__ float sdmem[NUM_WARP];
    float localmax = -INFINITY;

    // 各个线程从GMEM加载对应数据
    if (arridx < N) localmax = input[arridx];
    __syncthreads();

    // warp级归约, 求出warp区间的局部最大值
    localmax = warp_reduce_max(localmax);   // 最终只有warp中第一个线程存储着warp区间的局部最大值
    if (wCol == 0) sdmem[wRow] = localmax;
    __syncthreads();
    
    // threadblock级归约, 求出threadblock区间的局部最大值
    if (wRow == 0) {
        // 各个warp区间最大值保存在sdmem中, wRow == 0的warp中每个线程负责一个sdmem中一个值
        if (wCol < NUM_WARP) localmax = sdmem[wCol];
        else localmax = -INFINITY;
        localmax = warp_reduce_max(localmax);   // 最终只有warp中第一个线程存储着threadblock区间的局部最大值
        
        // grid级归约，求出全局区间的最大值
        if (wCol == 0) atomicMaxFloat(gmax, localmax);
    }
}

__device__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(FULL_MASK, v, 16);
    v += __shfl_down_sync(FULL_MASK, v, 8);
    v += __shfl_down_sync(FULL_MASK, v, 4);
    v += __shfl_down_sync(FULL_MASK, v, 2);
    v += __shfl_down_sync(FULL_MASK, v, 1);
    return v;
}

__global__ void reduce_sum_kernel(const float* input, const float* gmax, float* gsum, const int N) {
    const uint bidx = blockIdx.x;
    const uint tidx = threadIdx.x;
    const uint wRow = tidx / WARP_SIZE;
    const uint wCol = tidx % WARP_SIZE;
    const uint arridx = bidx * NUM_THREADS + tidx;

    __shared__ float sdmem[NUM_WARP];
    float localsum = 0.0;
    const float maxv = *gmax;

    // 各个线程从GMEM加载对应数据
    if (arridx < N) localsum += __expf(input[arridx] - maxv);
    __syncthreads();

    // warp级归约, 求出warp区间的局部和
    localsum = warp_reduce_sum(localsum);   // 最终只有warp中第一个线程存储着warp区间的局部和
    if (wCol == 0) sdmem[wRow] = localsum;
    __syncthreads();
    
    // threadblock级归约, 求出threadblock区间的局部最大值
    if (wRow == 0) {
        // 各个warp区间和保存在sdmem中, wRow == 0的warp中每个线程负责一个sdmem中一个值
        if (wCol < NUM_WARP) localsum = sdmem[wCol];
        else localsum = 0.0;
        localsum = warp_reduce_sum(localsum);   // 最终只有warp中第一个线程存储着threadblock区间的局部和
        
        // grid级归约，求出全局区间的最大值
        if (wCol == 0) atomicAdd(gsum, localsum);
    }
}

__global__ void softmax_kernel(const float* input, float* output, const float* gmax, const float* gsum, const int N) {
    const uint bidx = blockIdx.x;
    const uint tidx = threadIdx.x;
    const uint arridx = bidx * NUM_THREADS + tidx;
    const float maxv = *gmax;
    const float sumv = *gsum;

    if (arridx < N) output[arridx] = __expf(input[arridx] - maxv) / sumv;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    dim3 gridDim(CEIL_DIV(N, NUM_THREADS));
    dim3 blockDim(NUM_THREADS);
    
    float* d_max = nullptr;   // 全局内存, 用于同步各个threadblock, max(局部最大值)
    float* d_sum = nullptr;   // 全局内存, 用于同步各个threadblock, sum(局部总和)
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    
    // initialize d_max = -inf, d_sum = 0.0f
    float h_init_max = -INFINITY;
    float h_init_sum = 0.0f;
    cudaMemcpy(d_max, &h_init_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_init_sum, sizeof(float), cudaMemcpyHostToDevice);
    
    reduce_max_kernel<<<gridDim, blockDim>>>(input, d_max, N);
    reduce_sum_kernel<<<gridDim, blockDim>>>(input, d_max, d_sum, N);
    softmax_kernel<<<gridDim, blockDim>>>(input, output, d_max, d_sum, N);
    cudaDeviceSynchronize();
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void debug_reduce_max_kernel(const float* input, float* output, int N) {
    dim3 gridDim(CEIL_DIV(N, NUM_THREADS));
    dim3 blockDim(NUM_THREADS);
    
    float* d_max = nullptr;   // 全局内存, 用于同步各个threadblock, max(局部最大值)
    cudaMalloc(&d_max, sizeof(float));
    // initialize d_max = -inf, d_sum = 0.0f
    float h_init_max = -INFINITY;
    cudaMemcpy(d_max, &h_init_max, sizeof(float), cudaMemcpyHostToDevice);
    
    reduce_max_kernel<<<gridDim, blockDim>>>(input, d_max, N);

    cudaMemcpy(&h_init_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max: %f\n", h_init_max);
    cudaFree(d_max);
    
    cudaDeviceSynchronize();
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void debug_reduce_sum_kernel(const float* input, float* output, int N) {
    dim3 gridDim(CEIL_DIV(N, NUM_THREADS));
    dim3 blockDim(NUM_THREADS);
    
    float* d_max = nullptr;   // 全局内存, 用于同步各个threadblock, max(局部最大值)
    float* d_sum = nullptr;   // 全局内存, 用于同步各个threadblock, sum(局部总和)
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    
    // initialize d_max = -inf, d_sum = 0.0f
    float h_init_max = -INFINITY;
    float h_init_sum = 0.0f;
    cudaMemcpy(d_max, &h_init_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &h_init_sum, sizeof(float), cudaMemcpyHostToDevice);
    
    reduce_max_kernel<<<gridDim, blockDim>>>(input, d_max, N);
    reduce_sum_kernel<<<gridDim, blockDim>>>(input, d_max, d_sum, N);
    cudaMemcpy(&h_init_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum: %f\n", h_init_sum);
    cudaDeviceSynchronize();
}
