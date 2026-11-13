"""固定输入比较四种配置，并导出 PTX、SASS、资源及可复查的计时样本。

运行计时和导出：python analyze.py
仅供 Nsight Compute 捕获：python analyze.py --profile A
配置 A/B 为待解释的两个实现，C/D 每次只改变一个参数用于控制变量。

验证契约：四个 FP16 GEMM 必须匹配 torch.matmul；不改动被测 kernel。
证据：交替顺序计时、最终机器码和资源；计数器结果取决于驱动的 profiling 权限。
失效条件：源码、GPU、编译器、配置或竞争负载改变。
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import torch
import triton
import triton.testing

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from grouped_gemm import grouped_matmul_kernel

SPECS = {
    'A': (128, 128, 64, 8),
    'B': (256, 128, 32, 16),
    'C': (128, 128, 64, 16),
    'D': (128, 128, 32, 16),
}


def prepare():
    """四个相同尺寸 GEMM 共享一份计时输入，避免数据差异混入对照。"""
    torch.manual_seed(123)
    m, n, k = 1024, 8192, 8192
    matrices = []
    for _ in range(4):
        a = torch.rand((m, k), device='cuda', dtype=torch.float16)
        b = torch.rand((k, n), device='cuda', dtype=torch.float16)
        c = torch.empty((m, n), device='cuda', dtype=torch.float16)
        matrices.append((a, b, c, a @ b))

    ptrs = [torch.tensor([row[i].data_ptr() for row in matrices], device='cuda') for i in range(3)]
    metadata = [torch.tensor(values * 4, device='cuda', dtype=torch.int32)
                for values in [[m, n, k], [k, n, n]]]
    return matrices, (*ptrs, *metadata, 4)


def launch(name, args):
    """直接启动 JIT kernel，跳过 autotune，确保测到指定配置。"""
    bm, bn, bk, warps = SPECS[name]
    return grouped_matmul_kernel.fn[(128,)](
        *args,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, NUM_SM=128,
        num_warps=warps, num_stages=2,
    )


def check(matrices):
    """在计时区间之外比较独立参考结果。"""
    for _, _, actual, expected in matrices:
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def export(name, compiled):
    """最终 SASS 比 PTX 更适合判断 ptxas 引入的寄存器溢出位置。"""
    (OUT / f'{name}.ptx').write_text(compiled.asm['ptx'])
    (OUT / f'{name}.ttgir').write_text(compiled.asm['ttgir'])
    cubin = OUT / f'{name}.cubin'
    cubin.write_bytes(compiled.asm['cubin'])
    for tool, options, suffix in [
        ('cuobjdump', ['--dump-sass'], 'sass'),
        ('cuobjdump', ['--dump-resource-usage'], 'resources.txt'),
        ('nvdisasm', ['-g', '-c'], 'disasm'),
    ]:
        result = subprocess.run([tool, *options, str(cubin)], check=True, capture_output=True, text=True)
        (OUT / f'{name}.{suffix}').write_text(result.stdout)
    return dict(spec=SPECS[name], registers=compiled.n_regs,
                local_bytes=compiled.n_spills * 4, shared_bytes=compiled.metadata.shared)


def main():
    """分开普通计时和 profiler 捕获，避免把采集开销当作 kernel 性能。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=SPECS)
    options = parser.parse_args()
    matrices, args = prepare()
    if options.profile:
        # NOTE: 编译、预热和校验均在 profiler 区间外；仅捕获最后一次目标调用。
        launch(options.profile, args)
        check(matrices)
        for _ in range(20):
            launch(options.profile, args)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        launch(options.profile, args)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print('Profile launch complete:', options.profile, flush=True)
        return

    resources = {}
    for name in SPECS:
        compiled = launch(name, args)
        check(matrices)
        resources[name] = export(name, compiled)
        print(name, resources[name], flush=True)

    # NOTE: 先预热，再交替测量；do_bench 在各次调用前清理 L2，与现有 benchmark 一致。
    for _ in range(30):
        launch('A', args)
        launch('B', args)
    samples = []
    order_generator = random.Random(42)
    for round_id in range(7):
        names = list(SPECS)
        order_generator.shuffle(names)
        for name in names:
            median, p20, p80 = triton.testing.do_bench(
                lambda: launch(name, args), warmup=15, rep=80, quantiles=[0.5, 0.2, 0.8],
            )
            row = dict(round=round_id, name=name, median_ms=median, p20_ms=p20, p80_ms=p80)
            samples.append(row)
            print(row, flush=True)

    source = Path(grouped_matmul_kernel.fn.fn.__code__.co_filename)
    report = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                  shape=[1024, 8192, 8192], group_size=4, num_sm=128, num_stages=2,
                  kernel_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  resources=resources, samples=samples)
    (OUT / 'measurements.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
