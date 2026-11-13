"""汇总实测时间与 SASS 静态证据；不把逻辑请求量当作 DRAM 计数器值。

运行 analyze.py 后执行 python summarize.py。
适用范围为当前保存的四个 cubin，以及四个 1024x8192x8192 的完整 tile GEMM。
"""
from collections import Counter
import json
from pathlib import Path
import re
import statistics

OUT = Path(__file__).resolve().parent


def sass_evidence(name, resource):
    """以真实指令地址定位回跳，统计包含 HMMA 的最内层循环。"""
    instructions = []
    for lineno, line in enumerate((OUT / f'{name}.sass').read_text().splitlines(), 1):
        match = re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9.]*)\s+(.*?);', line)
        if match:
            instructions.append(dict(pc=int(match[1], 16), opcode=match[2], args=match[3], line=lineno))

    loops = []
    for ins in instructions:
        dest = re.search(r'0x([0-9a-f]+)', ins['args'])
        if ins['opcode'] == 'BRA' and dest and int(dest[1], 16) < ins['pc']:
            start, stop = int(dest[1], 16), ins['pc']
            body = [i for i in instructions if start <= i['pc'] <= stop]
            if any(i['opcode'].startswith('HMMA') for i in body):
                loops.append((start, stop, body))

    start, stop, body = min(loops, key=lambda loop: loop[1] - loop[0])
    local = [i for i in instructions if i['opcode'].startswith(('LDL', 'STL'))]
    inner_local = [i for i in local if start <= i['pc'] <= stop]
    bm, bn, bk, warps = resource['spec']
    tiles = 4 * (1024 // bm) * (8192 // bn)
    iterations = 8192 // bk
    counts = Counter(i['opcode'] for i in body)

    # NOTE: 当前输入各维完整，循环内 U16 加载的元素均有效；按线程计算逻辑请求字节数。
    assert all(i['opcode'] == 'LDG.E.U16' for i in body if i['opcode'].startswith('LDG'))
    logical_input_bytes = counts['LDG.E.U16'] * 2 * 32 * warps * iterations * tiles
    local_counts = Counter(i['opcode'] for i in local)
    local_bytes_per_tile_thread = sum(8 if '.64' in i['opcode'] else 4 for i in local)
    evidence = dict(
        k_loop_start=hex(start), k_loop_end=hex(stop), k_loop_sass_line=body[0]['line'],
        k_loop_static_instructions=len(body), k_loop_opcode_counts=dict(counts),
        total_output_tiles=tiles, k_iterations_per_tile=iterations,
        local_opcodes=dict(local_counts), local_ops_inside_k=len(inner_local),
        local_ops_outside_k=len(local) - len(inner_local),
        logical_input_read_bytes=logical_input_bytes,
        k_loop_warp_instruction_instances_estimate=len(body) * warps * iterations * tiles,
        source_A_read_bytes=4 * 1024 * 8192 * 8192 // bn * 2,
        source_B_read_bytes=4 * 1024 * 8192 * 8192 // bm * 2,
    )
    if name == 'B':
        # NOTE: B 的 local 指令全在每 tile 的循环体内、K 循环外，且均无谓词。
        assert not inner_local
        evidence['local_bytes_per_tile_thread_estimate'] = local_bytes_per_tile_thread
        evidence['logical_local_read_write_bytes_estimate'] = local_bytes_per_tile_thread * 32 * warps * tiles
    return evidence


report = json.loads((OUT / 'measurements.json').read_text())
summary = {}
for name, resource in report['resources'].items():
    times = [row['median_ms'] for row in report['samples'] if row['name'] == name]
    summary[name] = dict(resource=resource, median_ms=statistics.median(times),
                         min_round_median_ms=min(times), max_round_median_ms=max(times),
                         sass=sass_evidence(name, resource))

ratios = {}
for numerator, denominator in [('A', 'B'), ('A', 'C'), ('C', 'D'), ('D', 'B')]:
    paired = []
    for round_id in range(7):
        times = {row['name']: row['median_ms'] for row in report['samples'] if row['round'] == round_id}
        paired.append(times[numerator] / times[denominator])
    ratios[f'{numerator}/{denominator}'] = dict(median=statistics.median(paired), minimum=min(paired), maximum=max(paired))

result = dict(
    measured=summary, paired_time_ratios=ratios,
    counter_collection=dict(success=False, error='ERR_NVGPUCTRPERM', driver_setting='RmProfilingAdminOnly: 1',
                            unavailable=['DRAM traffic', 'cache hit rates', 'scheduler stalls', 'achieved occupancy', 'compute utilization']),
    interpretation=[
        'B 的 spill/refill 全在 K 循环外，并非每个 K 分块反复读写。',
        'D->B 只增大 BM，实测更快；新增 spill 的代价小于该配置变化带来的综合收益。',
        '逻辑输入读取量从 8 GiB 降到 6 GiB，来自 B 矩阵在更大 M tile 内的复用；这不是实测 DRAM 流量。',
        'A->C 单独增加 warp 未提速，不能把 B 的优势单独归因于更多 warp。',
        '实际缓存命中率、stall 原因及各因素的精确时间占比尚需具有权限的 Nsight Compute 采集。',
    ],
)
(OUT / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
for name, item in summary.items():
    evidence = item['sass']
    print(name, 'median_ms', item['median_ms'], 'K-loop local ops', evidence['local_ops_inside_k'],
          'logical input GiB', evidence['logical_input_read_bytes'] / 2**30)
print('B local logical MiB:', summary['B']['sass']['logical_local_read_write_bytes_estimate'] / 2**20)
