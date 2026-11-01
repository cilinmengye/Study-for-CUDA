import pytest
import torch
import triton
import triton.testing

from flash_attn import flash_attn_func

from flashattention_v1 import attention as attention_v1
from flashattention_v2 import attention as attention_v2

# 分别测试的是 BATCH, H, N_CTX, D_HEAD
TEST_SHAPES = [
    (1, 32, 128, 128),
    (2, 32, 256, 128),
    (4, 32, 512, 128),
]

PROVIDERS = ["triton-v1", "triton-v2", "flash"]

LINE_NAMES = {
    "triton-v1": "Triton v1",
    "triton-v2": "Triton v2",
    "flash": "FlashAttention",
}

STYLES = {
    "triton-v1": ("red", "-"),
    "triton-v2": ("green", "-"),
    "flash": ("blue", "-"),
}


def _make_inputs(Z, H, N_CTX, D_HEAD):
    torch.manual_seed(20)
    # Z is BATCH SIZE
    shape = (Z, H, N_CTX, D_HEAD)

    q = torch.empty(shape, device="cuda", dtype=torch.float16).normal_(0, 0.5)
    k = torch.empty_like(q).normal_(0, 0.5)
    v = torch.empty_like(q).normal_(0, 0.5)
    return q, k, v


def _make_fn(q, k, v, causal, sm_scale, provider):
    # sm_scale 为 QK^T/sqrt(d) 中的 1/sqrt(d), 其值可以不为 1/sqrt(d)
    if provider == "triton-v1":
        return lambda: attention_v1(q, k, v, causal, sm_scale)

    if provider == "triton-v2":
        return lambda: attention_v2(q, k, v, causal, sm_scale)

    if provider == "flash":
        # FlashAttention expects [BATCH, N_CTX, H, D_HEAD].
        # All layout conversions happen before timing.
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        return lambda: flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
        )

    raise ValueError(f"Unknown provider: {provider}")



# test_op，并通过装饰器记录测试参数。装饰器只添加测试信息，函数体在被调用时才执行。
# 或使用 pytest 启动测试 python -m pytest -v bench_flashattention.py
@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", TEST_SHAPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("provider", PROVIDERS)
@torch.no_grad()
def test_op(Z, H, N_CTX, D_HEAD, causal, provider):
    torch.set_float32_matmul_precision("highest")

    q, k, v = _make_inputs(Z, H, N_CTX, D_HEAD)
    sm_scale = D_HEAD ** -0.5

    # Independent FP32 reference implementation.
    p = torch.matmul(q.float(), k.float().transpose(2, 3)) * sm_scale

    if causal:
        M = torch.ones(
            (N_CTX, N_CTX),
            device=q.device,
            dtype=torch.bool,
        ).tril()
        p.masked_fill_(~M, -float("inf"))

    p = torch.softmax(p, dim=-1)
    ref_out = torch.matmul(p, v.float())

    out = _make_fn(q, k, v, causal, sm_scale, provider)()

    if provider == "flash":
        out = out.transpose(1, 2)

    torch.testing.assert_close(
        out.float(),
        ref_out,
        atol=1e-2,
        rtol=0,
    )


@torch.no_grad()
def run(providers):
    assert providers
    assert len(set(providers)) == len(providers)
    assert all(provider in PROVIDERS for provider in providers)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")


    # ====================== TEST 精度 ========================
    # Run correctness checks before collecting timings.
    for Z, H_TEST, N_CTX, D_HEAD_TEST in TEST_SHAPES:
        for causal in [False, True]:
            for provider in providers:
                test_op(
                    Z, H_TEST, N_CTX, D_HEAD_TEST,
                    causal, provider,
                )

    print("Forward correctness checks passed.")


    # ===================== TEST 性能 =========================
    # 测试不同 batch size 下，不同 num head 下, 不同 sequence len 下, 不同 head size 下
    # 在掩码是否启动下，测试性能。 causal 是否限制每个位置只能关注自身和之前的位置
    configs = []
    for BATCH in [16, 32]:
        for H in [32]:
            for D_HEAD in [64]:
                for causal in [True, False]:
                    configs.append(
                        triton.testing.Benchmark(
                            x_names=["N_CTX"],
                            x_vals=[2**i for i in range(10, 15)],
                            line_arg="provider",
                            line_vals=providers,
                            line_names=[LINE_NAMES[p] for p in providers],
                            styles=[STYLES[p] for p in providers],
                            ylabel="TFLOPS",
                            plot_name=(
                                f"attention-fwd-{'-'.join(providers)}"
                                f"-batch{BATCH}-head{H}-d{D_HEAD}-causal{causal}"
                            ),
                            args={
                            "BATCH": BATCH,
                            "H": H,
                            "D_HEAD": D_HEAD,
                            "causal": causal,
                            },
                        )
                    )

    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, provider):
        # N_CTX 为 sequence len
        # H 为 num head
        # D_HEAD 为 head size
        q, k, v = _make_inputs(BATCH, H, N_CTX, D_HEAD)
        sm_scale = D_HEAD ** -0.5
        fn = _make_fn(q, k, v, causal, sm_scale, provider)

        # Check the actual benchmark shape against FlashAttention.
        if provider != "flash":
            ref_out = _make_fn(
                q, k, v, causal, sm_scale, "flash"
            )().transpose(1, 2)

            torch.testing.assert_close(
                fn(),
                ref_out,
                atol=1e-2,
                rtol=0,
            )

        ms = triton.testing.do_bench(
            fn,
            warmup=25,
            rep=100,
            return_mode="mean",
        )

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * D_HEAD
        total_flops = 2 * flops_per_matmul

        if causal:
            total_flops *= 0.5

        # 求出 TFLOPS
        return total_flops * 1e-12 / (ms * 1e-3)


    bench_flash_attention.run(print_data=True)


if __name__ == "__main__":
    run(PROVIDERS)