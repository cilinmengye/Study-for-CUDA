import math
import torch
import triton
import triton.language as tl

ATOL = 1e-4 # 绝对误差容限
RTOL = 1e-4 # 相对误差容限

BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 32

# 每个元组的顺序为 (M, N, K)。
TEST_SHAPES = [
    (1, 1, 1),
    (35, 67, 19),
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (1003, 997, 509),
    (256, 768, 512),
]


# 需要 @triton.jit 才能通过 triton_gemm_kernel[grid]() 进行启动
@triton.jit
def triton_gemm_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for pid_k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        # 取出 a tiling 数据
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 取出 b tiling 数据
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 进行矩阵乘
        # "ieee" 要求使用 FP32 输入精度进行计算。官方文档给出的 NVIDIA 常见默认设置是 "tf32"
        accumulator += tl.dot(a_tile, b_tile, input_precision="ieee", out_dtype=tl.float32)
    
    # c_ptr.type.element_ty 获取输出矩阵 C 的元素数据类型：
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_gemm(
    a: torch.Tensor, 
    b: torch.Tensor, 
    c: torch.Tensor
):
    M, K = a.shape
    N = b.shape[1]
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N)
    )
    triton_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )


@torch.inference_mode()
def main():

    torch.cuda.set_device(0)

    for M, N, K in TEST_SHAPES:
        # 构造输入矩阵
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        a /= math.sqrt(K)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)
        # or c = torch.empty((M,N), device="cuda", dtype=torch.float32)
        c = torch.full((M, N), float("nan"), device="cuda", dtype=torch.float32)
        reference = torch.empty_like(c)

        def triton_fn():
            triton_gemm(a, b, c)
        
        def torch_fn():
            torch.mm(a, b, out=reference)
        
        triton_fn()
        torch_fn()
        torch.cuda.synchronize()

        error = torch.abs(c - reference)
        tolerence = ATOL + RTOL * torch.abs(reference)
        elementwise_close = error <= tolerence
        all_close = elementwise_close.all().item() 
        max_abs_error = (c - reference).abs().max().item()

        print(
            f"\nM={M}, N={N}, K={K}"
            f" correctness={'PASS' if all_close else 'FALSE'}"
            f" max_abs_error={max_abs_error:.6e}"
        )

if __name__ == "__main__":
    main()