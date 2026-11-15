"""
test_correctness.py：保留教程的四组 FP16 正确性检查。

运行：python test_correctness.py
使用 PyTorch 的 A @ B 作为参考；不支持 TMA 时只检查普通版本。
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

from common import DEVICE, is_cuda, supports_tma
from cublas_grouped_gemm import CublasGroupedGemm
from grouped_gemm import group_gemm_fn
from grouped_gemm_tma import group_gemm_tma_fn


def run_correctness_tests():
    """检查不同尺寸的 GEMM 能否在同一次 grouped kernel 调用中正确完成。"""
    # NOTE: 一组里有四个不同大小的 GEMM；各尺寸也满足普通 kernel 的完整 tile 假设。
    group_m = [1024, 512, 256, 128]
    group_n = [1024, 512, 256, 128]
    group_k = [1024, 512, 256, 128]
    group_A = []
    group_B = []
    group_B_T = []
    assert len(group_m) == len(group_n)
    assert len(group_n) == len(group_k)
    group_size = len(group_m)
    for i in range(group_size):
        M = group_m[i]
        N = group_n[i]
        K = group_k[i]
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)

        # NOTE: .T 只改变视图的形状和步长；contiguous() 才将 [N, K] 数据按行连续存储。
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)

    # NOTE: 两个实现使用相同的 A/B 数值；布局转换不会改变要验证的数学结果。
    tri_out = group_gemm_fn(group_A, group_B)
    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    for i in range(group_size):
        # NOTE: 浮点运算的累加次序可能不同，因此保留教程的绝对/相对误差容限。
        assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=1e-2)

    print(f"普通 Group GEMM：{group_size} 组 FP16 正确性检查通过。")

    if is_cuda():
        # NOTE: 一次 native grouped API 调用处理原教程的四种不同尺寸。
        with CublasGroupedGemm(group_A, group_B) as operation:
            cublas_out = operation.run()
            for expected, actual in zip(ref_out, cublas_out):
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        print(f"cuBLAS Grouped GEMM：{group_size} 组 FP16 正确性检查通过。")
        run_cublas_grouped_tests()

    if supports_tma():
        # NOTE: TMA 路径接收 B 的连续转置副本，参考结果仍然是原来的 A @ B。
        tri_tma_out = group_gemm_tma_fn(group_A, group_B_T)
        for i in range(group_size):
            assert torch.allclose(ref_out[i], tri_tma_out[i], atol=1e-2, rtol=1e-2)

        print(f"TMA Group GEMM：{group_size} 组 FP16 正确性检查通过。")
    else:
        print("跳过 TMA：教程要求 CUDA 设备计算能力至少为 9.0。")


def run_cublas_grouped_tests():
    """覆盖非方阵、重复分组、输出重用和非默认 CUDA stream。"""
    stream = torch.cuda.Stream(device=DEVICE)
    with torch.cuda.stream(stream):
        # NOTE: 重复形状不相邻，可验证分组重排后输出仍与原输入列表一一对应。
        shapes = [(31, 64, 96), (128, 80, 64), (31, 64, 96), (63, 128, 32)]
        group_A = [torch.randn((m, k), device=DEVICE, dtype=torch.float16) for m, n, k in shapes]
        group_B = [torch.randn((k, n), device=DEVICE, dtype=torch.float16) for m, n, k in shapes]
        group_C = [torch.full((m, n), float('nan'), device=DEVICE, dtype=torch.float16) for m, n, k in shapes]
        reference = [a @ b for a, b in zip(group_A, group_B)]

        with CublasGroupedGemm(group_A, group_B, group_C) as operation:
            assert operation.group_count == 3
            assert operation.group_sizes == [2, 1, 1]
            actual = operation.run()
            assert all(c is provided for c, provided in zip(actual, group_C))
            for c, expected in zip(actual, reference):
                torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)

            # NOTE: beta=0，每次都覆盖输出；输入数值改变后无需重新创建指针表。
            group_A[0].zero_()
            operation.run()
            torch.testing.assert_close(group_C[0], torch.zeros_like(group_C[0]), atol=0, rtol=0)
            for c, expected in zip(group_C[1:], reference[1:]):
                torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)

    print("cuBLAS Grouped GEMM：非方阵、重复分组、输出重用和非默认 stream 检查通过。")


if __name__ == "__main__":
    run_correctness_tests()
