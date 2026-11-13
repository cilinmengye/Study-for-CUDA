"""
benchmarks.py：保留教程的方阵、固定 N/K 两组性能对比。

运行：python benchmarks.py
无图形界面的终端可运行：MPLBACKEND=Agg python benchmarks.py
本文件将矩阵和元数据准备放在 Triton 计时区间之外，侧重 kernel 的执行时间。
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
import triton.testing

from common import DEVICE, num_sms, supports_tma
from grouped_gemm import grouped_matmul_kernel
from grouped_gemm_tma import grouped_matmul_tma_kernel, set_tma_allocator



# ===========================
# 计时区间内的调用
# ===========================

# only launch the kernel, no tensor preparation here to remove all overhead
def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, tuning_key):
    """使用已准备好的元数据启动普通 kernel。

    Args:
        a_ptrs: [Device] A 地址表。
        b_ptrs: [Device] B 地址表，B 为 [K, N]。
        c_ptrs: [Device] C 地址表。
        sizes: [Device] 展平的 M/N/K 尺寸表。
        lds: [Device] 展平的 A/B/C 行步长表。
        tuning_key: [Host] 尺寸和步长元组，在计时区间之外准备，用于区分调优缓存。
    """
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
        TUNING_KEY=tuning_key,
    )


def triton_tma_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size, dtype):
    """使用已准备好的元数据启动 TMA kernel；先由 run_benchmarks 注册分配器。

    Args:
        a_ptrs: [Device] A 地址表。
        b_ptrs: [Device] B 的连续转置副本地址表，布局为 [N, K]。
        c_ptrs: [Device] C 地址表。
        sizes: [Device] 展平的 M/N/K 尺寸表。
        lds: [Device] 展平的 A/转置后 B/C 行步长表。
    """
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_tma_kernel[grid](
        a_ptrs,
        b_ptrs,
        c_ptrs,
        sizes,
        lds,
        group_size,
        FP8=torch.float8_e4m3fn == dtype,
        NUM_SM=num_sms(),
    )


def torch_perf_fn(group_A, group_B):
    """沿用教程的基线，依次调用 PyTorch matmul 计算每个 GEMM。"""
    # NOTE: 这里保留 torch.matmul 自行创建输出的行为；Triton 路径则复用预分配的 C。
    for a, b in zip(group_A, group_B):
        torch.matmul(a, b)



# ===========================
# 两组性能测试
# ===========================

# NOTE: perf_report 定义横轴和各条曲线；真正执行测量的是末尾的 .run()。
@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['N'],
        x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'] + (['triton-tma'] if supports_tma() else []),
        # label name for the lines
        line_names=["cuBLAS", "Triton"] + (['Triton + TMA'] if supports_tma() else []),
        # line styles
        styles=[('green', '-'), ('blue', '-')] + ([('red', '-')] if supports_tma() else []),
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_square_matrices(N, provider):
    """固定每组四个 GEMM，改变方阵的边长 N。"""
    group_size = 4
    group_A = []
    group_B = []
    group_B_T = []
    A_addrs = []
    B_addrs = []
    B_T_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    # NOTE: 提前准备输入、输出及 B 的转置副本，避免把这些操作计入 Triton kernel 时间。
    for i in range(group_size):
        A = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
        B = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
        C = torch.empty((N, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        B_T_addrs.append(B_T.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [N, N, N]
        g_lds += [N, N, N]

    # NOTE: group_A/B/B_T/C 列表保留 tensor 引用，保证计时期间地址表中的显存有效。
    # 方阵 B 和 B_T 的行步长都是 N，因此这个测试可共用同一份 g_lds。
    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_b_t_ptrs = torch.tensor(B_T_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    # NOTE: 使用 CPU 端已有的元数据，避免为构造调优 key 从 GPU 读回尺寸。
    tuning_key = (tuple(g_sizes), tuple(g_lds))

    # NOTE: 返回耗时的中位数、20% 和 80% 分位数；min_ms/max_ms 是原教程变量名。
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(
                d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, tuning_key,
            ), quantiles=quantiles)
    if provider == 'triton-tma':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tma_perf_fn(d_a_ptrs, d_b_t_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, dtype=torch.
                                       float16), quantiles=quantiles)
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['M'],
        x_vals=sorted([2**i for i in range(5, 11)] + [31, 63, 127, 255, 511, 1023]),  # different possible values for `x_name`
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'] + (['triton-tma'] if supports_tma() else []),
        # label name for the lines
        line_names=["cuBLAS", "Triton"] + (['Triton + TMA'] if supports_tma() else []),
        # line styles
        styles=[('green', '-'), ('blue', '-')] + ([('red', '-')] if supports_tma() else []),
        ylabel="runtime(ms)",  # label name for the y-axis
        plot_name="group-gemm-performance-m-8192-k-8192",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark_batches(M, provider):
    """固定 N=K=8192、每组四个 GEMM，只改变 M；M 是矩阵行数，不是组内数量。"""
    N = 8192
    K = 8192
    group_size = 4
    group_A = []
    group_B = []
    group_B_T = []
    A_addrs = []
    B_addrs = []
    B_T_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    g_T_lds = []
    group_C = []
    for i in range(group_size):
        A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
        B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
        C = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
        B_T = B.T.contiguous()
        group_A.append(A)
        group_B.append(B)
        group_B_T.append(B_T)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        B_T_addrs.append(B_T.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]

        # NOTE: 普通 B 的行步长来自 [K, N] 布局，TMA 则必须使用 B_T 的 [N, K] 布局。
        # 本例 N=K 时数值碰巧相等，仍保留两份元数据以体现布局区别。
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]
        g_T_lds += [A.stride(0), B_T.stride(0), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
    d_b_t_ptrs = torch.tensor(B_T_addrs, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
    d_g_t_lds = torch.tensor(g_T_lds, dtype=torch.int32, device=DEVICE)

    # NOTE: M 改变时也重新选择配置，小 M 和大 M 无需共用同一个 tile。
    tuning_key = (tuple(g_sizes), tuple(g_lds))

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_perf_fn(
                d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, tuning_key,
            ), quantiles=quantiles)
    if provider == 'triton-tma':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_tma_perf_fn(d_a_ptrs, d_b_t_ptrs, d_c_ptrs, d_g_sizes, d_g_t_lds, group_size, dtype=torch.
                                       float16), quantiles=quantiles)
    return ms, min_ms, max_ms



# ===========================
# 独立运行入口
# ===========================

def run_benchmarks():
    """执行原有两组性能测试；独立运行时也初始化 TMA 所需的分配器。"""
    if supports_tma():
        # NOTE: 性能测试可单独运行，不再依赖正确性检查提前调用 TMA wrapper。
        set_tma_allocator()

    # NOTE: 普通版本现在按尺寸和步长分别调优，不再依赖之前先运行过哪个测试。
    benchmark_square_matrices.run(show_plots=True, print_data=True)
    benchmark_batches.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    run_benchmarks()
