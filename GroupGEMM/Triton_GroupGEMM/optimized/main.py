"""
main.py：独立优化副本的入口，先检查正确性，再运行性能测试。

本目录与上一级的学习代码独立；在 optimized 目录内执行下面的命令。
普通版本使用六组实测候选，按矩阵尺寸和步长分别调优，并修正 K 尾块 mask。
新增形状首次运行会重新调优；性能表不包含首次编译/调优耗时。

在当前目录运行（需要 GPU 版 PyTorch、Triton，以及绘图依赖 matplotlib/pandas）：
    python main.py                 # 运行全部内容，与原教程顺序一致
    python test_correctness.py     # 只检查正确性
    python benchmarks.py           # 只运行性能测试
无图形界面时可在命令前加 MPLBACKEND=Agg。首次运行会 JIT 编译并自动调优。

建议阅读顺序：
    test_correctness.py -> grouped_gemm.py -> grouped_gemm_tma.py -> benchmarks.py
    common.py 提供上述模块共享的设备信息。

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
