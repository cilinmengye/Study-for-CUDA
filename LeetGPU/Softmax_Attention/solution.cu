#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CEIL_DIV(M, N) (((M) +  (N - 1)) / (N))

template<const uint BM, const uint BN, const uint BK, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
matrixMulti_withTranspone_kernel(const float* A, const float* B, float* C, const int M, const int N, const int K)
{
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    const uint bRow = blockIdx.y;
    const uint bCol = blockIdx.x;
    const uint tRow = threadIdx.x / BN;
    const uint tCol = threadIdx.x % BN; 
    
    // 将 A 和 B 移动到正确位置
    A += bRow * BM * K;
    B += bCol * BN * K;
    C += bRow * BM * N + bCol * BN;

    float tmp = 0.0;
    for (int k = 0; k < K; k += BK) {
        // 将 A -> As
        if (bRow * BM + tRow < M && k + tCol < K) As[tRow * BK + tCol] = A[tRow * K + tCol];
        else As[tRow * BK + tCol] = 0.0;

        // 将 B -> Bs
        if (bCol * BN + tCol < N && k + tRow < K) Bs[tRow * BN + tCol] = B[tCol * K + tRow];
        else Bs[tRow * BN + tCol] = 0.0;
        __syncthreads();

        // 计算
        for (int bk = 0; bk < BK; bk++) tmp += As[tRow * BK + bk] * Bs[bk * BN + tCol];
        __syncthreads();

        // 移动 A 和 B
        A += BK;
        B += BK;
    }
    // 写回
    // ==dug== 因为没有写if (bRow * BM + tRow < M && bCol * BN + tCol < N)导致BUG
    if (bRow * BM + tRow < M && bCol * BN + tCol < N) C[tRow * N + tCol] = tmp;
}

template<const uint BM, const uint BN, const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
transpose_kernel(const float* A, float* AT, const uint M, const uint N)
{
    const uint bRow = blockIdx.y;
    const uint bCol = blockIdx.x;
    const uint tRow = threadIdx.x / BN;
    const uint tCol = threadIdx.x % BN;

    // 将 A 和 AT 移动到正确位置
    A += bRow * BM * N + bCol * BN;
    AT += bRow * BM + bCol * BN * M;

    if (bRow * BM + tRow < M && bCol * BN + tCol < N) AT[tCol * M + tRow] = A[tRow * N + tCol];
}

template<const uint NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
softmax_row_kernel(float* score, const uint M, const uint N, const float scale)
{
    __shared__ float tbmax[NUM_THREADS];
    const uint threadnum = blockDim.x;
    const uint tidx = threadIdx.x;
    const uint bidx = blockIdx.x;
    const uint row = bidx * N;
    
    //===========求每一行Max=========
    // 每个threadblock处理一行score矩阵的一行，得到除以sqrt(d)后的score值与一行的最大值
    float tmax = -INFINITY;
    for (int idx = tidx; idx < N; idx += threadnum) {
        float tmp = score[row + idx] * scale;
        score[row + idx] = tmp;
        tmax = fmaxf(tmax, tmp);
    }
    tbmax[tidx] = tmax;
    __syncthreads();

    // 进行threadblock级的reduce
    for (int idx = NUM_THREADS / 2; idx > 0; idx >>=1) {
        if (tidx < idx) {
            tmax = fmaxf(tmax, tbmax[tidx + idx]); 
            tbmax[tidx] = tmax;
        } 
        __syncthreads();
    }

    // 现在整个threadblock的最大值在tidx == 0的线程上, threadblock全部线程拿到这个最大值
    tmax = tbmax[0];

    //============求每一行Sum=============
    __shared__ float tbsum[NUM_THREADS];
    float tsum = 0.0;
    for (int idx = tidx; idx < N; idx += threadnum) {
        float tmp = expf(score[row + idx] - tmax);
        score[row + idx] = tmp;
        tsum += tmp;
    }
    tbsum[tidx] = tsum;
    __syncthreads();

    // 进行threadblock级的reduce
    for (int idx = NUM_THREADS / 2; idx > 0; idx >>=1) {
        if (tidx < idx) {
            tsum += tbsum[tidx + idx];
            tbsum[tidx] = tsum;
        }
        __syncthreads();
    }
    // 现在整个threadblock的sum在tidx == 0的线程上, threadblock全部线程拿到这个sum
    tsum = tbsum[0];

    //================求normalize====================
    for (int idx = tidx; idx < N; idx += threadnum) {
        float tmp = score[row + idx] / tsum;
        score[row + idx] = tmp;
    }
}

// print helper
extern void print_matrix(const std::vector<float>& A, int rows, int cols, const char* name);

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    const uint BM = 16;
    const uint BN = 16;
    const uint BK = 16;
    const uint NUM_THREADS = 256;

    float* vt = nullptr;
    float* score = nullptr;

    cudaMalloc(&vt, sizeof(float) * N * d);
    cudaMalloc(&score, sizeof(float) * M * N);

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim(NUM_THREADS);

    matrixMulti_withTranspone_kernel<BM, BN, BK, NUM_THREADS><<<gridDim, blockDim>>>(Q, K, score, M, N, d);
    // std::vector<float> score_h(M * N);
    // cudaMemcpy(score_h.data(), score, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    // print_matrix(score_h, M, N, "QK^T");
    
    // std::vector<float> v_h(N * d);
    // cudaMemcpy(v_h.data(), V, sizeof(float)*N*d, cudaMemcpyDeviceToHost);
    // print_matrix(v_h, N, d, "V");
    transpose_kernel<BN, BK, NUM_THREADS><<<gridDim, blockDim>>>(V, vt, N, d);   // 转置V可以让V在矩阵乘法访存时局部性更好
    // std::vector<float> vt_h(N * d);
    // cudaMemcpy(vt_h.data(), vt, sizeof(float)*N*d, cudaMemcpyDeviceToHost);
    // print_matrix(vt_h, d, N, "VT");
    
    float scale = 1.0f / sqrtf((float)d);
    softmax_row_kernel<NUM_THREADS / 2><<<M, NUM_THREADS / 2>>>(score, M, N, scale);
    // cudaMemcpy(score_h.data(), score, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
    // print_matrix(score_h, M, N, "softmax(QK^T/sqrt(d))");
    
    matrixMulti_withTranspone_kernel<BM, BN, BK, NUM_THREADS><<<gridDim, blockDim>>>(score, vt, output, M, d, N);

    cudaDeviceSynchronize();
}
