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

from common import DEVICE, supports_tma
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

    if supports_tma():
        # NOTE: TMA 路径接收 B 的连续转置副本，参考结果仍然是原来的 A @ B。
        tri_tma_out = group_gemm_tma_fn(group_A, group_B_T)
        for i in range(group_size):
            assert torch.allclose(ref_out[i], tri_tma_out[i], atol=1e-2, rtol=1e-2)

        print(f"TMA Group GEMM：{group_size} 组 FP16 正确性检查通过。")
    else:
        print("跳过 TMA：教程要求 CUDA 设备计算能力至少为 9.0。")


if __name__ == "__main__":
    run_correctness_tests()
