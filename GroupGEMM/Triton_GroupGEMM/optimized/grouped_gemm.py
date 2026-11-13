"""
grouped_gemm.py：普通 Group GEMM，使用显式指针加载矩阵分块。

建议先读 group_gemm_fn，了解 CPU 如何打包元数据，再读 grouped_matmul_kernel。
每组 GEMM 可以有不同的 M、N、K；使用 FP16，读取和写回均通过 mask 处理尾块。
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

import torch
import triton
import triton.language as tl

from common import DEVICE, num_sms



# ===========================
# Kernel 与自动调优
# ===========================

# PERF: 按组内矩阵的尺寸和步长分别调优，避免四个小 GEMM 的配置被四个大 GEMM 复用。
# NOTE: 尺寸表存储在 GPU 上，autotune 不会读取其中的值；Host 需要额外传入 TUNING_KEY。
# NOTE: 每行依次是 BM、BN、BK、warp 数；只保留实测有代表性的六组配置。
# 小 tile 服务小矩阵；较大 tile 配合更多 warp，分摊每线程的累加器压力。
# 当前通用 mask 路径中，增加 stages 没有稳定收益，因此统一使用 2。
grouped_gemm_configs = [
    triton.Config(
        {
            'BLOCK_SIZE_M': bm,
            'BLOCK_SIZE_N': bn,
            'BLOCK_SIZE_K': bk,
            'NUM_SM': num_sms(),
        },
        num_warps=warps,
        num_stages=2,
    )
    for bm, bn, bk, warps in [
        (64, 64, 32, 4),
        (32, 128, 32, 4),
        (32, 256, 64, 8),
        (64, 128, 64, 8),
        (128, 128, 64, 8),
        (256, 128, 32, 16),
    ]
]


@triton.autotune(
    configs=grouped_gemm_configs,
    key=['group_size', 'TUNING_KEY'],
)
@triton.jit
def grouped_matmul_kernel(
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
    TUNING_KEY: tl.constexpr = (),
):
    """用固定数量的 program 静态分配一组 GEMM 的全部输出 tile。

    Args:
        group_a_ptrs: [A_1_ptr, A_2_ptr, A_3_ptr ...] Input 矩阵地址表，注意 group_a_ptrs 本身也是个地址
        group_b_ptrs: [B_1_ptr, B_2_ptr, B_3_ptr ...] weight 矩阵地址表，注意 group_b_ptrs 本身也是个地址
        group_c_ptrs: [B_1_ptr, B_2_ptr, B_3_ptr ...] Output 矩阵地址表，注意 group_c_ptrs 本身也是个地址
        group_gemm_sizes: [[M_1, N_1_, K_1], [M_2, N_2, K_2], ...] 每个 group 矩阵 shape 信息
        g_lds: [[stride_AM_1, stride_BK_1, stride_CM_1], [stride_AM_2, stride_BK_2, stride_CM_2], ...] 每个 group 矩阵的 stride 信息
        TUNING_KEY: [Host] 尺寸/步长元组，仅区分调优缓存，不参与 GPU 数值计算。
    """
    # 采用的是 persistent GEMM 的思路：启动固定数量的 program，让它们在 kernel 内部循环完成全部 tile。

    # NOTE: 把所有 GEMM 的输出 tile 串成一条全局序列。
    # program p 依次处理 p、p + NUM_SM、p + 2 * NUM_SM……，一个 program 可处理多个 tile。
    # NUM_SM 是启动的 program 数量，不表示把某个 program 绑定到某个物理 SM。
    tile_idx = tl.program_id(0)

    # last_problem_end 记录当前参与 GEMM 计算的 expert 起始 tile.
    # num_tile_1 = tl.cdiv(M_1, BLOCK_M) * tl.cdiv(N_1, BLOCK_N); 计算到 expert 1 时，last_problem_end = 0
    # num_tile_2 = tl.cdiv(M_2, BLOCK_M) * tl.cdiv(N_2, BLOCK_N); 计算到 expert 2 时，last_problem_end += num_tile_1
    # ...
    last_problem_end = 0


    # 对于每个 group, 即 expert, group 分组的单位是 expert
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        # NOTE: 只划分输出的 M/N 维；K 是每个输出 tile 内部需要完整累加的维度。
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # NOTE: 当前 GEMM 占据 [last_problem_end, last_problem_end + num_tiles)。
        # tile_idx 落入这个区间才计算；否则继续寻找所属的下一个 GEMM。
        # iterate through the tiles in the current gemm problem
        # 对于这个矩阵运算，我们采用 persistent GEMM 的思路：启动固定数量的 program，让它们在 kernel 内部循环完成全部 tile。
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)

            # NOTE: 先从地址表读取整数地址，再转成 FP16 指针，才能访问实际矩阵。
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            # NOTE: 减去前缀 tile 数，得到当前矩阵内部的编号；按行优先还原二维坐标。
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # NOTE: 地址 = 基址 + 行号 * 行步长 + 列号；[:, None] 与 [None, :] 广播成二维网格。
            # A 分块为 [BLOCK_SIZE_M, BLOCK_SIZE_K]，B 分块为 [BLOCK_SIZE_K, BLOCK_SIZE_N]。
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]

            # NOTE: 用 FP32 累加 K 方向各分块的乘积，最后再转回 FP16，减少累加误差。
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # NOTE: 加载前按本轮 K 坐标重新计算 mask；offs_k 保持为块内偏移。
                # 若在加载后用 kk 更新 mask，下一轮会使用滞后的边界条件。
                current_k = offs_k + kk * BLOCK_SIZE_K
                a_mask = (offs_am[:, None] < gm) & (current_k[None, :] < gk)
                b_mask = (current_k[:, None] < gk) & (offs_bn[None, :] < gn)

                # NOTE: multiple_of 向编译器提供对齐信息，不会自动对齐地址或检查越界。
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                # NOTE: 原教程假设完整 tile；这里用本轮 mask 给无效位置补零。
                # assume full tile for now
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                accumulator += tl.dot(a, b)

                # NOTE: 沿 K 前进：A 向右移动若干列，B 向下移动若干行。
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb

            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            # NOTE: my add
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)

            # assumes full tile for now
            tl.store(c_ptrs, c, mask=c_mask)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # NOTE: 切换 GEMM 时只推进区间起点，不重置 tile_idx，才能保持全局分工不重不漏。
        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles



# ===========================
# Host 端调用
# ===========================

def group_gemm_fn(group_A, group_B):
    """计算一组 C[i] = A[i] @ B[i]，返回设备 tensor 列表。

    Args:
        group_A: [Host] 非空 Python 列表，元素为 [Device] FP16 tensor，形状 [M, K]。
        group_B: [Host] Python 列表，元素为 [Device] FP16 tensor，形状 [K, N]。

    两组 tensor 须位于 DEVICE，且最后一维连续。M/N/K 的尾块通过 mask 处理。

    Example:
        group_A: [A_1, A_2, A_3 ...], A_i is torch.Tensor((M, K))
        group_B: [B_1, B_2, B_3 ...], B_i is torch.Tensor((K, N))

    """
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    # NOTE: Python 列表不能直接交给 kernel；先为每个问题收集地址、尺寸和行步长。
    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape

        # NOTE: group_C 持有输出 tensor，保证 kernel 使用裸地址时显存仍然有效。
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())

        # NOTE: 两张元数据表实际是一维 tensor，每三个元素描述一个 GEMM。
        # stride(0) 表示跨一行需要跳过的元素数，不是字节数；末维步长隐含为 1。
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # NOTE: 拷贝到 GPU 的是地址和元数据；A/B 矩阵内容仍留在原来的显存中。
    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    # NOTE: grid 与问题的 tile 总数无关；kernel 内的循环会持续计算静态分配给自己的 tile。
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        TUNING_KEY=(tuple(g_sizes), tuple(g_lds)),
    )

    return group_C
