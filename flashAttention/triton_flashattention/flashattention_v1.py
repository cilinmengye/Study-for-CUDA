"""Forward adapted from the historical Triton fused-attention tutorial."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M, Out,
    stride_qh, stride_qm, stride_qk,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MODE: tl.constexpr,
):
    # TODO
    pass

def attention(q, k, v, causal=False, sm_scale=None):
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape
    assert q.device == k.device == v.device and q.is_cuda
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert not (q.requires_grad or k.requires_grad or v.requires_grad)

    # Z is Batch size
    # N_CTX is sequence len
    Z, H, N_CTX, D_HEAD = q.shape
    assert Z > 0 and H > 0 and N_CTX > 0 and N_CTX % 128 == 0
    assert D_HEAD in {16, 32, 64, 128}

    if sm_scale is None:
        sm_scale = D_HEAD ** -0.5
    assert 0 < sm_scale < float("inf")

    o = torch.empty_like(q)
    L = torch.empty((Z * H, N_CTX), device=q.device, dtype=torch.float32)
    M = torch.empty_like(L)


    grid = (Z, H)
    num_warps = 4 if D_HEAD <= 64 else 8

    if causal:
        modes = [1] if N_CTX <= 2048 else [2, 3]
    else:
        modes = [0]

    with torch.cuda.device(q.device):
        for mode in modes:
            _fwd_kernel[grid](
                q, k, v, sm_scale,
                L, M, o,
                q.stride(1), q.stride(2), q.stride(3),
                N_CTX=N_CTX,
                BLOCK_M=128,
                BLOCK_DMODEL=D_HEAD,
                BLOCK_N=64,
                MODE=mode,
                num_warps=num_warps,
                num_stages=2,
            )

    return o


if __name__ == "__main__":
    from bench_flashattention import run

    run(["triton-v1", "flash"])