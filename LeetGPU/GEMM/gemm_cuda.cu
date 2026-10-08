#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>    // std::printf、std::fprintf、stderr
#include <cstdlib>   // std::exit、EXIT_FAILURE
#include <algorithm> // std::max
#include <cmath>
#include <vector>
#include <random>

// 在宏定义内部，# 会把参数对应的代码文本转换成字符串字面量。
#define CUDA_CHECK(expression)  \
    do {    \
        const cudaError_t status = (expression);    \
        if (status != cudaSuccess) {    \
            std::fprintf(    \
                stderr, "%s:%d :%s failed: %s\n",   \
                __FILE__, __LINE__, #expression,    \
                cudaGetErrorString(status));    \
            std::exit(EXIT_FAILURE); \
        }   \
    } while(0)

#define CUBLAS_CHECK(expression)    \
    do {    \
        const cublasStatus_t status = (expression);  \
        if (status != CUBLAS_STATUS_SUCCESS) {  \
            std::fprintf(   \
                stderr, "%s:%d: %s failed: status=%d\n",  \
                __FILE__, __LINE__, #expression,    \
                static_cast<int>(status));  \
            std::exit(EXIT_FAILURE);    \
        }   \
    } while(0)

#define C_DIV(M, N) (((M) + (N) - 1) / (N))

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 32;
constexpr int NUM_THREADS = 128;
constexpr double ATOL = 1e-4;
constexpr double RTOL = 1e-4;

// __global__ 表示这个函数在 GPU 上执行，由 CPU 启动。
// const float* 表示通过指针读取输入矩阵。
// __launch_bounds__(NUM_THREADS) 提供线程数量等约束，影响编译器的寄存器使用策略
template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
cuda_gemm_kernel(
    const float* a,
    const float* b,
    float* c,
    int M,
    int N,
    int K
) {
    // constexpr 表明这个数值在编译时确定
    // 在正常优化编译下，编译器通常能够消除 stride_block_am = K 这样的别名
    // 并复用相同的 N、K 数值。
    const int stride_block_am = K; 
    constexpr int stride_block_ak = 1;
    const int stride_block_bk = N; 
    constexpr int stride_block_bn = 1;
    const int stride_block_cm = N;
    constexpr int stride_block_cn = 1;
    
    // BLOCK 中的 Tread 维度
    int tid = threadIdx.x;

    // BLOCK 维度
    int bid_m = blockIdx.y;
    int bid_n = blockIdx.x;
    int boffs_m = bid_m * BLOCK_M;
    int boffs_n = bid_n * BLOCK_N;

    // Tiling 共享内存
    __shared__ float a_tile[BLOCK_M][BLOCK_K];
    __shared__ float b_tile[BLOCK_K][BLOCK_N];
    
    // 每个 Thread 的内部累加存储
    constexpr int TILE_ELEMENTS = BLOCK_M * BLOCK_N;
    constexpr int OUTPUTS_PER_THREAD = C_DIV(TILE_ELEMENTS, NUM_THREADS);
    float accumulator[OUTPUTS_PER_THREAD] = {0.0f};

    int ktile_num = C_DIV(K, BLOCK_K);
    for (int bid_k = 0; bid_k < ktile_num; bid_k++) {
        int boffs_k = bid_k * BLOCK_K;
        //==== 我们联合 block 的 thread 将 Tiling 数据放入 share memory 上 ====

        // 所有线程协作读取 A 的一个分块
        // 在 block 维度上移动
        // 此处注意语法问题：a + offset 的类型仍然是 const float*
        // 因此需要使用 const float* 而不是 float*
        const float* a_block_ptr = 
            a + (boffs_m * stride_block_am + boffs_k * stride_block_ak);

        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += NUM_THREADS) {
            // 在 tile 维度上移动
            int tile_m = idx / BLOCK_K;
            int tile_k = idx % BLOCK_K;
            
            // mask
            bool a_mask = ((boffs_m + tile_m) < M) & ((boffs_k + tile_k) < K);
            if (a_mask) {
                const float* a_ptr = a_block_ptr + 
                    (tile_m * stride_block_am + tile_k * stride_block_ak);
                
                a_tile[tile_m][tile_k] = a_ptr[0];   
            }
            else {
                a_tile[tile_m][tile_k] = 0.0f;
            }
        }

        // 所有线程协作读取 B 的一个分块
        // 在 block 维度上移动
        const float* b_block_ptr = 
            b + (boffs_k * stride_block_bk + boffs_n * stride_block_bn);
        
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += NUM_THREADS) {
            // 在 thread 维度上移动
            int tile_k = idx / BLOCK_N;
            int tile_n = idx % BLOCK_N;
            
            // mask
            bool b_mask = ((boffs_k + tile_k) < K) & ((boffs_n + tile_n) < N);
            if (b_mask) {
                const float* b_ptr = b_block_ptr + 
                    (tile_k * stride_block_bk + tile_n * stride_block_bn);
                
                b_tile[tile_k][tile_n] = b_ptr[0];
            } else {
                b_tile[tile_k][tile_n] = 0.0f;
            }
        }

        // 等待全部将完整数据写入共享内存
        __syncthreads();

        // 进行矩阵乘计算
        // 每个 thread 负责 c tiling 矩阵上一个元素
        #pragma unroll // 展开固定次数的循环，方便编译器处理累加数组。
        for (int slot = 0; slot < OUTPUTS_PER_THREAD; slot++) {
            int idx = tid + NUM_THREADS * slot;

            if (idx < TILE_ELEMENTS) {
                int tile_m = idx / BLOCK_N;
                int tile_n = idx % BLOCK_N;
                
                for (int tile_k = 0; tile_k < BLOCK_K; tile_k++) {
                    accumulator[slot] += 
                        a_tile[tile_m][tile_k] * 
                        b_tile[tile_k][tile_n];
                }
            }
        }

        // 所有线程完成后，才能加载下一个 K 分块。
        __syncthreads();
    }

    // 写入最终的结果
    // c 在 block 维度上的移动
    float* c_block_ptr = c + 
        (boffs_m * stride_block_cm + boffs_n * stride_block_cn);

    #pragma unroll
    for (int slot = 0; slot < OUTPUTS_PER_THREAD; slot++) {
        int idx = tid + slot * NUM_THREADS;
        
        if (idx < TILE_ELEMENTS) {
            int tile_m = idx / BLOCK_N;
            int tile_n = idx % BLOCK_N;
            
            bool c_mask = ((boffs_m + tile_m) < M) & ((boffs_n + tile_n) < N);
            if (c_mask) {
                // c 在 thread 维度上的移动.
                float* c_ptr = c_block_ptr +
                    (tile_m * stride_block_cm + tile_n * stride_block_cn);
                c_ptr[0] = accumulator[slot];
            }
        }
    }
}


void cuda_gemm(
    const float* a,
    const float* b,
    float* c,
    int M,
    int N,
    int K
) {
    // dim3 用于表示 CUDA 的三维尺寸，省略的维度默认为 1。
    // 注意这里和 Triton 反过来了，在 CUDA 中参数填写顺序为 (x,y,z)
    // Triton 中是 (y,x,z)

    dim3 block(NUM_THREADS);    // 一维 block，每个 block 有 NUM_THREADS 个线程。
    dim3 grid(
        (N + BLOCK_N - 1) / BLOCK_N,
        (M + BLOCK_M - 1) / BLOCK_M
    );      // grid.x 遍历 N 方向，grid.y 遍历 M 方向。
    cuda_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_THREADS>
        <<<grid, block>>>(a, b, c, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}

// 使用 cuBLAS 计算参考结果。
void cublas_gemm(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int M,
    int N,
    int K
) {
    // cuBLAS GEMM 计算：C = alpha * A * B + beta * C。
    float alpha = 1.0f;
    float beta = 0.0f;

    // cublasSgemm 中的 S 表示使用 FP32。
    //
    // cuBLAS 按列优先解释矩阵。
    // 我们的行优先内存，按列优先解释时对应转置矩阵。
    // 因此通过计算 C^T = B^T * A^T 得到需要的结果。
    //
    // 参数先传 b，再传 a；尺寸依次传 N、M、K。
    // CUBLAS_OP_N 表示 cuBLAS 不额外转置输入。
    //
    // N、K、N 是三个矩阵在列优先视角下的列间距。
    // &alpha 和 &beta 表示传入这两个变量的地址。
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b, N,
        a, K,
        &beta,
        c, N
    ));
}


int main() {
    CUDA_CHECK(cudaSetDevice(0));

    // handle 保存 cuBLAS 的计算环境，后续调用需要传入它。
    // &handle 让 cublasCreate 将创建结果写入这个变量。
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // 要求使用 FP32 精度，便于和自己的 FP32 kernel 比较。
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    // 每一行保存一个 (M, N, K)。
    const int TEST_SHAPES[][3] = {
        {1, 1, 1},
        {35, 67, 19},
        {128, 128, 128},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {1003, 997, 509},
        {256, 768, 512},
    };
    int num_shapes = sizeof(TEST_SHAPES) / sizeof(TEST_SHAPES[0]);

    // C++ 标准库的随机数生成器。
    // 0 是随机种子；distribution 生成均值 0、标准差 1 的正态随机数。
    std::mt19937 generator(0);
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    for (int test = 0; test < num_shapes; test++) {
        int M = TEST_SHAPES[test][0];
        int N = TEST_SHAPES[test][1];
        int K = TEST_SHAPES[test][2];

        // 矩阵元素个数
        size_t count_a = static_cast<size_t>(M) * K;
        size_t count_b = static_cast<size_t>(K) * N;
        size_t count_c = static_cast<size_t>(M) * N;

        // 矩阵所需内存 bytes 大小
        size_t bytes_a = count_a * sizeof(float);
        size_t bytes_b = count_b * sizeof(float);
        size_t bytes_c = count_c * sizeof(float);

        // 分配 GPU 指针和地址
        float* g_a = nullptr;
        float* g_b = nullptr;
        float* g_c = nullptr;
        float* g_reference = nullptr;

        CUDA_CHECK(cudaMalloc(&g_a, bytes_a));
        CUDA_CHECK(cudaMalloc(&g_b, bytes_b));
        CUDA_CHECK(cudaMalloc(&g_c, bytes_c));
        CUDA_CHECK(cudaMalloc(&g_reference, bytes_c));

        // 我们不能直接在 GPU 所指向的内存上分配数据，而是需要通过 CPU 上准备输入数据，然后复制到 GPU
        // 所以我们还需要创建 CPU 指针和地址
        std::vector<float> c_a(count_a);
        std::vector<float> c_b(count_b);
        std::vector<float> c_c(count_c);
        std::vector<float> c_reference(count_c);
        // 初始化数值
        float scale = 1.0f / std::sqrt(static_cast<float>(K));
        for (size_t i = 0; i < count_a; i++) {
            c_a[i] = distribution(generator) * scale;
        }
        for (size_t i = 0; i < count_b; i++) {
            c_b[i] = distribution(generator);
        }

        // 将数值 copy 到 GPU 地址上
        // .data() 是它的成员函数，返回指向内部数组第一个元素的指针
        CUDA_CHECK(cudaMemcpy(
            g_a, c_a.data(), bytes_a, cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            g_b, c_b.data(), bytes_b, cudaMemcpyHostToDevice 
        ));

        // 启动函数执行 GEMM
        cuda_gemm(g_a, g_b, g_c, M, N, K);
        cublas_gemm(handle, g_a, g_b, g_reference, M, N, K);

        // 等待 GPU 执行完成
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 将数据拷贝回 CPU
        CUDA_CHECK(cudaMemcpy(
            c_c.data(), g_c, bytes_c, cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaMemcpy(
            c_reference.data(), g_reference, bytes_c, cudaMemcpyDeviceToHost 
        ));

        // 检查精度
        bool all_close = true;
        float max_abs_error = 0.0f; 
        for (size_t i = 0; i < count_c; i++) {
            float error = std::abs(c_c[i] - c_reference[i]);
            float tolerance = ATOL + RTOL * std::abs(c_reference[i]);

            if (!(error <= tolerance)) {
                all_close = false;
            }
            max_abs_error = std::max(max_abs_error, error);
        }

        std::printf(
            "\nM=%d, N=%d, K=%d correctness=%s max_abs_error=%.6e\n",
            M, N, K,
            all_close ? "PASS" : "FALSE",
            max_abs_error
        );


        // 释放掉 GPU 内存
        CUDA_CHECK(cudaFree(g_a));
        CUDA_CHECK(cudaFree(g_b));
        CUDA_CHECK(cudaFree(g_c));
        CUDA_CHECK(cudaFree(g_reference));

        if (!all_close) {
            std::exit(EXIT_FAILURE);
        }
    }

    //末尾需要释放 cuBLAS handle。
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return 0;
}