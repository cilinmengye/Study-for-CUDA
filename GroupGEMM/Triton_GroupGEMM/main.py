"""
main.py：Group GEMM 学习入口，先检查正确性，再运行性能测试。

在当前目录运行（需要 GPU 版 PyTorch、Triton，以及绘图依赖 matplotlib/pandas）：
    python main.py                 # 运行全部内容，与原教程顺序一致
    python test_correctness.py     # 只检查正确性
    python benchmarks.py           # 只运行性能测试
    python cublas_grouped_gemm.py   # 单独运行 cuBLAS 原生 Grouped GEMM 示例
无图形界面时可在命令前加 MPLBACKEND=Agg。首次运行会 JIT 编译并自动调优。

建议阅读顺序：
    test_correctness.py -> grouped_gemm.py -> grouped_gemm_tma.py -> benchmarks.py
    common.py 提供上述模块共享的设备信息。

性能表中的 PyTorch loop 是逐次 torch.matmul；cuBLAS Grouped 则直接调用
cublasGemmGroupedBatchedEx，输出/指针表/handle 均在计时前准备，使用 FP32 累加。

教程来源：https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
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

from benchmarks import run_benchmarks
from test_correctness import run_correctness_tests


def main():
    """按教程顺序运行；导入本模块时不会自动启动测试或性能测量。"""
    run_correctness_tests()
    run_benchmarks()


if __name__ == "__main__":
    main()
