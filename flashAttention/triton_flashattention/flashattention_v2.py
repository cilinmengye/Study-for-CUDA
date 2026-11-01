"""Forward adapted from the Triton FlashAttention-2 tutorial."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    offs_m, offs_n,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    # [lo, hi) 表示 KV Cache token 能够处理的范围。
    if STAGE == 1:
        # Blocks strictly before the diagonal block.
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Diagonal block, requiring a causal mask.
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # Entire non-causal attention.
        lo, hi = 0, N_CTX

    # tl.advance(block_ptr, offsets): 按逻辑维度移动 block pointer，返回移动后的指针。
    # offsets 中每个数值表示沿对应维度移动多少个元素。实际地址偏移会根据创建 block pointer 时提供的 strides 计算。
    # 第一个维度保持不变，第二个维度前进 lo, 因为 K 的逻辑矩阵形状是转置的
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # 第一个维度前进 lo，第二个维度保持不变
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        # tl.multiple_of 向编译器提供整除关系的提示，返回值保持原来的数值。
        # 可以理解为向编译器声明: start_n % BLOCK_M == 0
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # K_block_ptr 的 block_shape=(D_HEAD, BLOCK_N), 所以 
        # load 的时候取出来的也是 shape = (D_HEAD, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * qk_scale

        # 处理 casual 第二个阶段
        if STAGE == 2:
            # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_n = tl.arange(0, BLOCK_N)
            # 即 offs_m 表示此次分块 Q 不同的 token 的计数下标
            # start_n 表示此次处理 KV Cache 的初始范围下标;
            # start_n + offs_m[None, :] 则此次要处理的 KV Cache token 计数下标
            # mask 的 shape = (BLOCK_M, BLOCK_N), 对于分块 Q 中每个token 若计数下标 
            # 大于等于 KV Cache token 计数下标，则元素为 True, 否则为 False
            mask = offs_m[:, None] >= start_n + offs_n[None, :]
            # tl.where 令 mask 对应位置为 True：保留原来的 qk 数值。
            # mask 对应位置为 False：将该位置设置为负无穷 -inf。
            qk = tl.where(mask, qk, -float("inf"))

        # 这里专门分出 causal 第一阶段和第二阶段是为了减少非矩阵计算。

        # Use the running maximum directly.
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)

        # V2 keeps an unnormalized output accumulator.
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(tl.float16), v, acc)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, Out,
    stride_qh, stride_qm, stride_qk,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)
    qvk_offset = off_hz * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_qk, stride_qm),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)

    # 我们讨论 casual = True
    # 对于此次分块处理的 Q, 其行坐标范围为 [start_m*BLOCK_M, (start_m+1)*BLOCK_M)
    # 而分块 Q 中的 query token i, 其 KV Cache 允许处理的范围为 [0, i+1).
    # 在分块 Q 中有许多不同的 query token, 所以 KV Cache 允许处理的范围是不同的。
    # 但是 KV Cache 允许处理的范围边界为 [start_m*BLOCK_M, (start_m+1)*BLOCK_M) 
    # casual_masking 当 casual = True 时, 将 casual masking 当成两个阶段处理
    # 第一个阶段, 分块 Q 中的 token 一定都能够处理的 KV Cache 范围: [0, start_m*BLOCK_M)
    # 第二个阶段，分块 Q 中的 token 不一定都能够处理的 KV Cache 范围: [start_m*BLOCK_M, (start_m+1)*BLOCK_M)

    # 当我们不开启 casual 时， STAGE = 1 会执行第一个分支，且分支 STAGE = 3, 即处理全部 KV Cache 长度
    # 当我们开启 casual 时， STAGE = 3 会执行第一个分支，且分支 STAGE = 1, 即处理上述我们说的第一个阶段
    # 同时还会执行第二个分支，且分支 STAGE = 2, 即处理上述我们说的第二个阶段

    # causal=False: STAGE=1, inner STAGE=3 processes all keys.
    # causal=True:  STAGE=3, inner STAGE=1 processes earlier blocks.
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            offs_m, offs_n,
            N_CTX,
            BLOCK_M, BLOCK_N,
            4 - STAGE,
        )

    # causal=True: process the diagonal block in the same kernel.
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            offs_m, offs_n,
            N_CTX,
            BLOCK_M, BLOCK_N,
            2,
        )

    # Normalize only once, after processing all K/V blocks.
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(tl.float16))


def attention(q, k, v, causal=False, sm_scale=None):
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape
    assert q.device == k.device == v.device and q.is_cuda
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert not (q.requires_grad or k.requires_grad or v.requires_grad)

    Z, H, N_CTX, D_HEAD = q.shape
    assert Z > 0 and H > 0 and N_CTX > 0 and N_CTX % 128 == 0
    assert D_HEAD in {16, 32, 64, 128}

    if sm_scale is None:
        sm_scale = D_HEAD ** -0.5
    assert 0 < sm_scale < float("inf")

    o = torch.empty_like(q)
    grid = (triton.cdiv(N_CTX, 128), Z * H)

    # 较大的 D_HEAD 会增加每个 program 的计算量和需要保存的数据量。
    # 当 D_HEAD 增大时，让更多 thread 分担计算与中间数据，有机会减轻每个 thread 的寄存器压力。
    num_warps = 4 if D_HEAD <= 64 else 8

    with torch.cuda.device(q.device):
        # num_warps 和 num_stages 由 Triton 的启动器与编译器读取，因此无需在 _attn_fwd 函数体中显式引用。
        # num_warps 指定一个 program 使用多少个 warp 协作执行。
        # 对于这里的普通 NVIDIA Triton kernel，一个 program 对应一个 CTA：
        # num_warps=4  # 每个 CTA 使用 4 × 32 = 128 个 thread
        # num_warps=8  # 每个 CTA 使用 8 × 32 = 256 个 thread
        # num_stages=2 配置循环的软件流水线深度，用于尝试重叠数据加载与计算。
        _attn_fwd[grid](
            q, k, v, sm_scale, o,
            q.stride(1), q.stride(2), q.stride(3),
            N_CTX=N_CTX,
            HEAD_DIM=D_HEAD,
            BLOCK_M=128,
            BLOCK_N=64,
            STAGE=3 if causal else 1,
            num_warps=num_warps,
            num_stages=2,
        )

    return o


if __name__ == "__main__":
    from bench_flashattention import run

    run(["triton-v2", "flash"])