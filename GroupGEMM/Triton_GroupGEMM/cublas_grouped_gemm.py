"""cublas_grouped_gemm.py：直接调用 cuBLAS Grouped GEMM 的 FP16 学习示例。

运行：python cublas_grouped_gemm.py
依赖 GPU 版 PyTorch 和带 cublasGemmGroupedBatchedEx 的 cuBLAS（12.5+）；
通过标准库 ctypes 调用 C API，不需要安装 nvmath-python 或编译 C++ 扩展。

用法：with CublasGroupedGemm(group_A, group_B) as operation: operation.run()
构造时准备输出、分组元数据和 GPU 指针表；run() 仅提交一次 grouped API 调用。
对象绑定创建时的 CUDA stream，须在该设备/stream 下使用；退出 with 时等待完成。
本示例仅处理连续、行主序、二维 FP16 CUDA tensor，不提供 autograd。
"""

import ctypes as ct
from functools import lru_cache

import torch

# NOTE: 与 CUDA/cuBLAS 头文件中的枚举一致，数值不是 tensor 的 dtype 编号。
CUDA_R_16F = 2
CUBLAS_COMPUTE_32F = 68


def _check(status, operation):
    """检查 native API 的返回值，避免把失败调用当作有效的性能结果。"""
    if status != 0:
        raise RuntimeError(f"{operation} failed: cuBLAS status={status}")


@lru_cache(maxsize=1)
def _load_cublas():
    """绑定与当前 PyTorch CUDA 主版本对应的已安装 cuBLAS 动态库。"""
    if torch.version.cuda is None:
        raise RuntimeError("此示例需要 NVIDIA CUDA 版 PyTorch。")

    library = ct.CDLL(f"libcublas.so.{torch.version.cuda.split('.')[0]}")
    if not hasattr(library, 'cublasGemmGroupedBatchedEx'):
        raise RuntimeError("cuBLAS 缺少 cublasGemmGroupedBatchedEx；需要 cuBLAS 12.5+。")

    # NOTE: 显式声明指针参数，防止 ctypes 把 64 位地址按默认的 32 位整数传递。
    pointer = ct.c_void_p
    integer = ct.c_int
    int_pointer = ct.POINTER(integer)
    library.cublasCreate_v2.argtypes = [ct.POINTER(pointer)]
    library.cublasDestroy_v2.argtypes = [pointer]
    library.cublasSetStream_v2.argtypes = [pointer, pointer]
    library.cublasGetVersion_v2.argtypes = [pointer, int_pointer]
    library.cublasGemmGroupedBatchedEx.argtypes = [
        pointer, int_pointer, int_pointer, int_pointer, int_pointer, int_pointer,
        pointer, pointer, integer, int_pointer, pointer, integer, int_pointer,
        pointer, pointer, integer, int_pointer, integer, int_pointer, integer,
    ]
    for name in [
        'cublasCreate_v2', 'cublasDestroy_v2', 'cublasSetStream_v2',
        'cublasGetVersion_v2', 'cublasGemmGroupedBatchedEx',
    ]:
        getattr(library, name).restype = integer

    return library



# ===========================
# 准备元数据与重复执行
# ===========================

class CublasGroupedGemm:
    """持有矩阵、指针表和独立 cuBLAS handle，重复计算 C[i] = A[i] @ B[i]。"""

    def __init__(self, group_A, group_B, group_C=None):
        """在计时前完成分组、分配和 handle 初始化。

        Args:
            group_A: [Host] 列表，元素为 [Device] 连续 FP16 矩阵，形状 [M, K]。
            group_B: [Host] 列表，元素为 [Device] 连续 FP16 矩阵，形状 [K, N]。
            group_C: [Host] 可选列表，元素为 [Device] 连续 FP16 输出矩阵 [M, N]。
                输出之间、输出与输入之间不得重叠；省略时自动分配。
        """
        if not group_A or len(group_A) != len(group_B):
            raise ValueError("A/B 列表必须非空且长度相同。")
        if group_C is not None and len(group_C) != len(group_A):
            raise ValueError("C 列表长度必须与 A/B 相同。")

        self.device = group_A[0].device
        self.group_A, self.group_B = list(group_A), list(group_B)
        self.group_C = [] if group_C is None else list(group_C)
        groups = {}
        for index, (a, b) in enumerate(zip(self.group_A, self.group_B)):
            for tensor in (a, b):
                if (not tensor.is_cuda or tensor.device != self.device or
                        tensor.dtype != torch.float16 or tensor.ndim != 2 or not tensor.is_contiguous()):
                    raise ValueError("输入必须为同一 GPU 上的连续二维 FP16 tensor。")
            m, k = a.shape
            if b.shape[0] != k or min(m, k, b.shape[1]) <= 0:
                raise ValueError("矩阵尺寸必须为正，且 A.shape[1] == B.shape[0]。")
            n = b.shape[1]

            if group_C is None:
                self.group_C.append(torch.empty((m, n), device=self.device, dtype=torch.float16))
            c = self.group_C[index]
            if (c.device != self.device or c.dtype != torch.float16 or
                    c.shape != (m, n) or not c.is_contiguous()):
                raise ValueError("输出必须为对应形状的连续 FP16 CUDA tensor。")

            # NOTE: 同形状 GEMM 归入一组；指针表按组重排，输出列表仍保留原始顺序。
            groups.setdefault((m, n, k), []).append(index)

        self.group_count = len(groups)
        self.group_sizes = [len(indices) for indices in groups.values()]
        order = [index for indices in groups.values() for index in indices]
        shapes = list(groups)
        ints = ct.c_int * self.group_count
        floats = ct.c_float * self.group_count

        # NOTE: cuBLAS 按列主序解释数据。计算 C^T = B^T @ A^T，交换 A/B 和 M/N，
        # 即可直接使用 PyTorch 的行主序内存；不需要转置或复制矩阵内容。
        # 下列尺寸、步长、缩放系数和 group_size 都是 [Host] 数组。
        self.trans = ints(*([0] * self.group_count))  # CUBLAS_OP_N
        self.m = ints(*(n for m, n, k in shapes))
        self.n = ints(*(m for m, n, k in shapes))
        self.k = ints(*(k for m, n, k in shapes))
        self.sizes = ints(*self.group_sizes)
        self.alpha = floats(*([1.0] * self.group_count))
        self.beta = floats(*([0.0] * self.group_count))

        with torch.cuda.device(self.device):
            # NOTE: 指针表本身也是 [Device] tensor；原矩阵引用由当前对象持有。
            self.stream = torch.cuda.current_stream(self.device)
            self.a_ptrs = torch.tensor([self.group_B[i].data_ptr() for i in order],
                                       dtype=torch.int64, device=self.device)
            self.b_ptrs = torch.tensor([self.group_A[i].data_ptr() for i in order],
                                       dtype=torch.int64, device=self.device)
            self.c_ptrs = torch.tensor([self.group_C[i].data_ptr() for i in order],
                                       dtype=torch.int64, device=self.device)
            self.library = _load_cublas()
            self.handle = ct.c_void_p()
            _check(self.library.cublasCreate_v2(ct.byref(self.handle)), 'cublasCreate')
            try:
                # NOTE: 独立 handle 避免改变 PyTorch 自己使用的 cuBLAS 状态。
                _check(self.library.cublasSetStream_v2(self.handle, self.stream.cuda_stream), 'cublasSetStream')
                version = ct.c_int()
                _check(self.library.cublasGetVersion_v2(self.handle, ct.byref(version)), 'cublasGetVersion')
                self.version = version.value
            except Exception:
                self.close()
                raise

        # NOTE: 行主序连续矩阵交换后，列主序的 lda=N、ldb=K、ldc=N。
        # 新 handle 默认使用 HOST pointer mode，正好对应 CPU 上的 alpha/beta 数组。
        self.arguments = (
            self.handle, self.trans, self.trans, self.m, self.n, self.k,
            self.alpha, self.a_ptrs.data_ptr(), CUDA_R_16F, self.m,
            self.b_ptrs.data_ptr(), CUDA_R_16F, self.k,
            self.beta, self.c_ptrs.data_ptr(), CUDA_R_16F, self.m,
            self.group_count, self.sizes, CUBLAS_COMPUTE_32F,
        )


    def run(self):
        """在绑定的 stream 上提交一次 grouped API 调用；不分配 tensor、不主动同步 GPU。"""
        if not self.handle.value:
            raise RuntimeError("cuBLAS operation 已关闭。")

        _check(self.library.cublasGemmGroupedBatchedEx(*self.arguments), 'cublasGemmGroupedBatchedEx')
        return self.group_C


    def close(self):
        """在计时区间外等待完成，再释放 handle；可重复调用。"""
        if self.handle.value:
            # WARNING: 确保 kernel 不再使用裸指针后，调用者才能释放矩阵和指针表。
            self.stream.synchronize()
            _check(self.library.cublasDestroy_v2(self.handle), 'cublasDestroy')
            self.handle.value = None


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()



# ===========================
# 独立运行示例
# ===========================

def main():
    """同时演示不同形状分组、同形状合并和原始输出顺序。"""
    shapes = [(31, 64, 96), (128, 80, 64), (31, 64, 96)]
    group_A = [torch.randn((m, k), device='cuda', dtype=torch.float16) for m, n, k in shapes]
    group_B = [torch.randn((k, n), device='cuda', dtype=torch.float16) for m, n, k in shapes]
    with CublasGroupedGemm(group_A, group_B) as operation:
        actual = operation.run()
        for a, b, c in zip(group_A, group_B, actual):
            torch.testing.assert_close(c, a @ b, atol=1e-2, rtol=1e-2)
        print(f'cuBLAS {operation.version}: group_count={operation.group_count}, group_size={operation.group_sizes}')
        print('cuBLAS Grouped GEMM：不同形状及重复形状检查通过。')


if __name__ == '__main__':
    main()
