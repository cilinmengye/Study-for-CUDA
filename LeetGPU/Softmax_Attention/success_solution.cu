#include <cuda_runtime.h>
#include <math.h>

#define TILE 16

// 1. 矩阵转置：将 K (Nxd) 转置为 KT (dxN)
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < cols && r < rows) {
        output[c * rows + r] = input[r * cols + c];
    }
}

// 2. 分块矩阵乘法：C = A * B
__global__ void matrixMulti_kernel(const float* A, const float* B, float* C, int M, int N, int K_dim) {
    __shared__ float AS[TILE][TILE];
    __shared__ float BS[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K_dim + TILE - 1) / TILE; ++t) {
        // 加载 Tile 到共享内存
        if (row < M && (t * TILE + threadIdx.x) < K_dim)
            AS[threadIdx.y][threadIdx.x] = A[row * K_dim + t * TILE + threadIdx.x];
        else
            AS[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE + threadIdx.y) < K_dim)
            BS[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            BS[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) sum += AS[threadIdx.y][k] * BS[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

// 3. 数值稳定的行级 Softmax

__global__ void softmax_row_kernel(float* score, int M, int N, float scale) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    __shared__ float sdata[256];

    // 1. max
    float max_val = -1e20f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = score[row * N + j] * scale;
        score[row * N + j] = v;
        max_val = fmaxf(max_val, v);
    }

    sdata[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float row_max = sdata[0];

    // 2. sum
    float sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = expf(score[row * N + j] - row_max);
        score[row * N + j] = v;
        sum += v;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float row_sum = sdata[0] + 1e-6f;

    // 3. normalize
    for (int j = tid; j < N; j += blockDim.x) {
        score[row * N + j] /= row_sum;
    }
}


// 4. 解决入口
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *d_KT, *d_Score;
    cudaMalloc(&d_KT, sizeof(float) * d * N);
    cudaMalloc(&d_Score, sizeof(float) * M * N);

    dim3 block(TILE, TILE);

    // Step 1: K -> K^T (dxN)
    dim3 grid_trans((d + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    transpose_kernel<<<grid_trans, block>>>(K, d_KT, N, d);

    // Step 2: Score = Q * K^T (MxN)
    dim3 grid_score((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matrixMulti_kernel<<<grid_score, block>>>(Q, d_KT, d_Score, M, N, d);

    // Step 3: Softmax(Score / sqrt(d))
    float scale = 1.0f / sqrtf((float)d);
    softmax_row_kernel<<<M, 256>>>(d_Score, M, N, scale);

    // Step 4: Output = Score * V (Mxd)
    dim3 grid_out((d + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matrixMulti_kernel<<<grid_out, block>>>(d_Score, V, output, M, d, N);

    cudaFree(d_KT);
    cudaFree(d_Score);
}