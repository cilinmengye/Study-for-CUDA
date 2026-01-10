#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B, float *As, float *Bs,
                             int innerRowA, int innerColA, int innerRowB, int innerColB) {
    
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        // (offset + innerRowA) 表示加载到数组的对应行上; * BK 和 * N 是因为As和A处于不同的维度的数组中
        // innerColA * 4 因为我们一次取4个元素所以需要乘以4
        reinterpret_cast<float4 *>(&As[(offset + innerRowA) * BK + innerColA * 4])[0] = 
        reinterpret_cast<const float4 *>(&A[(offset + innerRowA) * N + innerColA * 4])[0];
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int TK>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, float *As, float *Bs, 
                const uint warpRow, const uint warpCol, const uint threadRowInWarp, const uint threadColInWarp) {
    // 将As和Bs的位置偏移置正确位置
    // (warpRow * WM) * TK 移动到thread对应的warp位置上; threadRowInWarp * TM 移动到thread位置上
    As += (warpRow * WM) * TK + threadRowInWarp * TM;
    // warpCol * WN 移动到thread对应的warp位置上; threadColInWarp * TN 移动到thread位置上
    Bs += warpCol * WN + threadColInWarp * TN;
    // 线程处理块的外循环
    for (uint bkIdx = 0; bkIdx < BK; bkIdx += TK) {
        // 从Share Memory中的As, Bs加载每个线程各自所需的数据到regM, regN
        // 加载As -> regM
        for (uint thRowIdx = 0; thRowIdx < TM; thRowIdx++){
            // 我们一次取4个元素
            for (uint thColIdx = 0; thColIdx < TK; thColIdx += 4) {
                reinterpret_cast<float4 *>(&regM[thRowIdx * TK + thColIdx])[0] = 
                reinterpret_cast<const float4 *>(&As[thRowIdx * BK + thColIdx])[0];
            }
        }
        // 加载Bs -> regN
        for (uint thRowIdx = 0; thRowIdx < TK; thRowIdx++) {
            for (uint thColIdx = 0; thColIdx < TN; thColIdx += 4) {
                reinterpret_cast<float4 *>(&regN[thRowIdx * TN + thColIdx])[0] = 
                reinterpret_cast<const float4 *>(&Bs[thRowIdx * BN + thColIdx])[0];
            }
        }
        // 点积
        for (uint thMIdx = 0; thMIdx < TM; thMIdx++) {
            for (uint thKIdx = 0; thKIdx < TK; thKIdx++) {
                for (uint thNIdx = 0; thNIdx < TN; thNIdx++) {
                    threadResults[thMIdx * TN + thNIdx] += 
                    regM[thMIdx * TK + thKIdx] * regN[thKIdx * TN + thNIdx];
                }
            }
        }
        // 更新As和Bs位置
        As += TK;
        Bs += TK * BN;
    }
}

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 * @tparam TK The per-thread tile size for K dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int TK, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    // Placement of the threadblock in mutrix tile
    const uint cRow = blockIdx.y;     // [0, 31]
    const uint cCol = blockIdx.x;     // [0, 31]

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE;  // the warp this thread is in; WARPSIZE = 32 and max threadIdx.x = 127 so value in range [0, 3]
    const uint warpCol = warpIdx % (BN / WN);     // BN / WN 表示在Col方向上需要的warp数; BN = 128; WN = 64; [0, 1]
    const uint warpRow = warpIdx / (BN / WN);     // BN = 128; WN = 64; [0, 1]

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;      // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WN / TN); // WN / TN 表示在Col方向需要的thread数; WN = 64； TN = 8; [0, 7]
    const uint threadRowInWarp = threadIdxInWarp / (WN / TN); // WN = 64； TN = 8; [0, 3]

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    // 注意此处C的位置我们已经考虑到了warp层次了
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step 
    const int innerRowA = threadIdx.x / (BK / 4);     // BK / 4 表示在Col方向需要的thread数; BK = 32; [0, 15]
    const int innerColA = threadIdx.x % (BK / 4);     // [0, 7]
    const int rowStrideA = (NUM_THREADS) / (BK / 4);  // NUM_THREADS = 128; rowStrideA = 16 表示经过一次threadblock全体执行后, 下次再执行数组需要位移的行数
    const int innerRowB = threadIdx.x / (BN / 4);     // BN = 128; [0, 3]
    const int innerColB = threadIdx.x % (BN / 4);     // [0, 31]
    const int rowStrideB = (NUM_THREADS) / (BN / 4);  // NUM_THREADS = 128; rowStrideB = 4

    // allocate thread-local cache for results in registerfile
    float threadResults[TM * TN] = {0.0};
    // we cache into registers on the warptile level
    float regM[TM * TK] = {0.0};
    float regN[TK * TN] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();

        wt::processFromSmem<BM, BN, BK, WM, WN, TM, TN, TK>(
            regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
        
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    // 将C移动到对应位置，注意我们是在C已经偏移到Warp层面位置上继续将其移动到thread层面
    float *C_interim = C + (threadRowInWarp * TM) * N + (threadColInWarp * TN);
    for (uint thRowIdx = 0; thRowIdx < TM; thRowIdx++) {
        for (uint thColIdx = 0; thColIdx < TN; thColIdx += 4) {
            float4 tmp = reinterpret_cast<float4 *>(&C_interim[thRowIdx * N + thColIdx])[0];
            const int i = thRowIdx * TN + thColIdx;
            tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
            // write back
            reinterpret_cast<float4 *>(&C_interim[thRowIdx * N + thColIdx])[0] = tmp;
        }
    }
}