"""
grouped_gemm_tma.py：TMA Group GEMM，使用 tensor descriptor 加载和保存分块。

调度方式与普通版本相同，主要区别是数据搬运方式，以及 B 按 [N, K] 连续存储。
保留教程的 FP16/FP8 路径；是否执行由 common.supports_tma 判断。
"""

# Copyright (c) 2023 - 2025 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Optional

import torch
import triton
import triton.language as tl

from common import DEVICE, num_sms, supports_tma



# ===========================
# TMA Kernel 与自动调优
# ===========================

# PERF: 保留原来的 16 组候选；除 tile 大小外，也比较流水线阶段数和每个 CTA 的 warp 数。
tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K' : BK}, num_stages=s, num_warps=w) \
    for BM in [128]\
    for BN in [128, 256]\
    for BK in [64, 128]\
    for s in ([3, 4])\
    for w in [4, 8]\
]


@triton.autotune(
    tma_configs,
    key=['group_size'],
)
@triton.jit
def grouped_matmul_tma_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # is the output FP8 or FP16
    FP8: tl.constexpr,
):
    """沿用普通版本的全局 tile 调度，以 descriptor 完成矩阵分块读写。

    Args:
        group_a_ptrs: [Device] A 矩阵地址表，矩阵布局为 [M, K]。
        group_b_ptrs: [Device] 已转置并连续存储的 B 地址表，矩阵布局为 [N, K]。
        group_c_ptrs: [Device] C 矩阵地址表，矩阵布局为 [M, N]。
        group_gemm_sizes: [Device] 按 M、N、K 顺序展平的尺寸表。
        g_lds: [Device] 按 A、转置后 B、C 顺序展平的行步长表，单位为元素。
    """
    # NOTE: FP8 同时决定输入指针类型和输出类型；中间累加始终使用 FP32。
    dtype = tl.float8e4nv if FP8 else tl.float16

    # NOTE: 与普通版本相同，program p 负责全局 tile p、p + NUM_SM、p + 2 * NUM_SM……。
    tile_idx = tl.program_id(0)
    last_problem_end = 0

    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # NOTE: 只有当前 program 有这个 GEMM 的任务时才创建 descriptor。
        # descriptor 放在 while 外，同一 program 处理该 GEMM 的多个 tile 时可以复用。
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # pick up a tile from the current gemm problem
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            # NOTE: descriptor 描述整张矩阵的基址、形状、步长，以及一次搬运的分块形状。
            # 后面的 load/store 只需给出分块左上角坐标，无需手动构造每个元素的地址。
            a_desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[gm, gk],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            # NOTE: B 按 [N, K] 存储，让 K 维连续，加载的分块为 [BLOCK_SIZE_N, BLOCK_SIZE_K]。
            b_desc = tl.make_tensor_descriptor(
                b_ptr,
                shape=[gn, gk],
                strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
            )

            c_desc = tl.make_tensor_descriptor(
                c_ptr,
                shape=[gm, gn],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            # iterate through the tiles in the current gemm problem
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                k = gk

                # NOTE: 从全局 tile 编号恢复当前 GEMM 内部的二维输出 tile 坐标。
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # do regular gemm here
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    # NOTE: descriptor 的偏移量以元素为单位，kk 每次沿 K 前进一个分块。
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])

                    # NOTE: b.T 把已加载的 [BN, BK] 分块转为 [BK, BN]，恢复 A @ B 的数学含义。
                    accumulator += tl.dot(a, b.T)

                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N

                c = accumulator.to(dtype)
                # NOTE: descriptor 带有矩阵形状，分块读写由 descriptor 处理边界。
                c_desc.store([offs_cm, offs_cn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # NOTE: 只推进 GEMM 的区间起点；tile_idx 跨 GEMM 保持连续。
        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles



# ===========================
# Host 端调用与显存分配
# ===========================

def set_tma_allocator():
    """注册教程中的 descriptor 分配器，供正常调用和独立性能测试共用。"""
    # NOTE: kernel 内创建 descriptor 需要全局显存；回调由 Triton 在启动时调用。
    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


def group_gemm_tma_fn(group_A, group_B):
    """计算一组 A @ B；这里 group_B 接收的是数学意义上 B 的转置副本。

    Args:
        group_A: [Host] 非空列表，元素为 [Device] tensor，形状 [M, K]。
        group_B: [Host] 列表，元素为 [Device] tensor，形状 [N, K]；传 B.T.contiguous()。

    输入须位于 DEVICE，使用一致的 FP16 或 float8_e4m3fn 类型，且最后一维连续。
    TMA 要求基址按 16 字节对齐、行步长对应的字节数是 16 的倍数。
    """

    assert supports_tma()

    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    # NOTE: 与普通版本一样打包地址、尺寸和步长，区别是 B 的物理布局为 [N, K]。
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[1]
        M, K = A.shape
        N, K = B.shape

        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())

        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    set_tma_allocator()

    # NOTE: 这里 NUM_SM 固定取设备的 SM 数，autotune 只搜索 tile、num_stages 和 num_warps。
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_tma_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        FP8=torch.float8_e4m3fn == group_A[0].dtype,
        NUM_SM=num_sms(),
    )

    return group_C
