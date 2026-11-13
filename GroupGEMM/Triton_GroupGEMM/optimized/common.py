"""
common.py：共享设备信息。

沿用教程的设备选择和硬件能力判断；导入时需要可用的 GPU 后端。
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

# NOTE: 输入、输出和指针表都放在同一个活动设备上，kernel 才能直接访问它们。
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    """区分 CUDA 与其他 Triton 后端，供硬件能力判断使用。"""
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    """沿用教程条件：CUDA 后端且计算能力主版本至少为 9。"""
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def num_sms():
    """返回教程使用的调度规模；非 CUDA 后端保留原来的默认值。"""
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count

    return 148
