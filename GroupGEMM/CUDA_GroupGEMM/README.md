# CUDA Group GEMM 学习基线

独立的 CUDA C++ 项目：cuBLAS 原生 Grouped GEMM、CUTLASS Grouped GEMM、自定义 SIMT Group GEMM，以及共同的正确性/性能测试。支持 FP16、BF16 输入输出，统一使用 FP32 累加。

## 目录与阅读顺序

```text
CUDA_GroupGEMM/
├── CMakeLists.txt                  # 构建与 CTest 入口
├── README.md
├── .gitignore
├── include/                        # 公共接口与共用资源管理
│   ├── grouped_gemm.h
│   └── custom_gemm.h               # 自定义 kernel 的两个 CUDA 入口
├── src/                            # 实际调用库的实现
│   ├── cublas_grouped_gemm.cu
│   ├── cutlass_grouped_gemm.cu
│   ├── cutsom_gemm.cu              # 两种布局调度与 tile_gemm 入口
│   └── tilegemm/
│       └── tilegemm_v0.cu          # 原有 tile GEMM 计算核心
├── tests/                          # 数据生成、参考值和正确性断言
│   ├── testbed.h
│   ├── test_correctness.cu
│   └── test_custom_gemm.cu          # custom 边界、Graph 与动态 valid_rows 检查
├── benchmarks/                     # 配置筛选、计时、CSV 输出
│   ├── benchmarks.cu              # main：资源准备与场景遍历
│   ├── benchmark_options.h
│   ├── benchmark_options.cu        # 命令行、结果路径与帮助信息
│   ├── benchmark_timing.h         # Stream / BatchTimer，含 CUDA Graph
│   ├── benchmark_timing.cu        # 预热、统计与多轮计时
│   ├── custom_operation.h         # custom 后端的参数与配置适配
│   ├── benchmark_runner.h
│   └── benchmark_runner.cu        # 边界检查、选优、正式计时与 CSV
├── results/                        # 自动生成的 CSV，Git 忽略
│   └── results.csv
└── build/                          # 编译产物与 CTest 日志，Git 忽略
```

建议按下面的顺序阅读，核心逻辑旁已添加中文解释和具体索引示例：

1. [include/grouped_gemm.h](include/grouped_gemm.h)：先看 `GroupedProblem`，理解有效行数、物理容量和专家起点；再看设备内存的生命周期。
2. [tests/test_correctness.cu](tests/test_correctness.cu)：看 `Dataset` 如何产生同一份 GPU 输入，以及独立参考如何检查输出。
3. [src/cublas_grouped_gemm.cu](src/cublas_grouped_gemm.cu)：理解两层地址表、`Yᵀ = W × Aᵀ` 和 cuBLAS 的 group 定义。
4. [src/cutlass_grouped_gemm.cu](src/cutlass_grouped_gemm.cu)：理解 CTA/warp/MMA 三层 tile、流水线、设备端调度和 `initialize/run` 的职责。
5. [benchmarks/benchmarks.cu](benchmarks/benchmarks.cu)：从 `main()` 顺着 `run_case()` 看完整流程，再看 `BatchTimer` 的 Event/Graph 计时。

阅读自定义 kernel 时，先看 [include/custom_gemm.h](include/custom_gemm.h) 中两种布局的元素地址公式，再看 [src/cutsom_gemm.cu](src/cutsom_gemm.cu) 中共用的 `tile_gemm()` 和两个外层调度 kernel。

拆分后，`benchmarks.cu` 只保留 `main()`；原函数体、默认参数、计时顺序和注释随职责迁移，阅读位置如下：

| 职责 | 文件与原有符号 |
|---|---|
| 命令行与结果路径 | [benchmark_options.h](benchmarks/benchmark_options.h) 的 `Options`；[benchmark_options.cu](benchmarks/benchmark_options.cu) 的 `parse_options/print_help/wants` |
| CUDA stream、Graph 和计时 | [benchmark_timing.h](benchmarks/benchmark_timing.h) 的 `Stream/BatchTimer`；[benchmark_timing.cu](benchmarks/benchmark_timing.cu) 的 `warm_up/median/measure_rounds` |
| custom 配置适配 | [custom_operation.h](benchmarks/custom_operation.h) 的 `CustomOperation` |
| 场景与调优 | [benchmark_runner.cu](benchmarks/benchmark_runner.cu) 的 `check_edges/tune_candidates/run_case` |

## 构建与运行

需要 CUDA Toolkit（包含 `cublasGemmGroupedBatchedEx`）、CMake 3.24+ 和 CUTLASS 头文件。当前构建目标为 `sm_89`，用于 RTX 4090。CUTLASS 使用 SM80 Tensor Core 指令路径，无需 Hopper/TMA。

在本目录下，本机可以直接运行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR=/mnt/hdd/users/yxlin/python-envs/flashinfer-cu130/lib/python3.12/site-packages/flashinfer/data/cutlass
cmake --build build -j 2
./build/group_gemm_bench
```

默认输出到本项目的 `results/results.csv`。程序自动创建缺失的结果目录；从 `build/` 或其他目录启动也写入同一个项目 `results/`。输出根目录由 CMake 配置时记录，移动整个项目后需要重新配置和构建。

`--csv` 的**相对路径以项目 `results/` 为基准**：`--csv decode_bf16.csv` 写入 `results/decode_bf16.csv`，`--csv experiment/run.csv` 写入 `results/experiment/run.csv`。绝对路径按指定位置输出。`--help` 和 `--check-only` 不创建 CSV 或结果目录。整个 `results/` 已加入 `.gitignore`，运行结果不随 Git 发布。

在其他机器上，将 `CUTLASS_DIR` 换成包含 `include/cutlass` 的 CUTLASS 根目录。这里只使用 FlashInfer 附带的头文件，程序不依赖 Python、PyTorch 或 FlashInfer 的运行时。默认链接系统 CUDA Toolkit 的 cuBLAS，版本可能与 Python 环境里的 cuBLAS 不同。

常用命令：

```bash
# 仅做正确性检查：两个主场景、所有候选和边界案例，不调优、不生成 CSV。
./build/group_gemm_bench --check-only
ctest --test-dir build --output-on-failure

# 只比较 Decode / BF16 的 Graph 表现。
./build/group_gemm_bench --scenario decode --dtype bf16 --timing graph --csv decode_bf16.csv

# 只运行 cuBLAS；可用 --device 1 切换到第二张卡。
./build/group_gemm_bench --provider cublas --device 0 --csv cublas.csv

# 检查 GPU 非法访问；此运行的耗时不能当作性能数据。
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/group_gemm_bench --check-only
```

所有选项见 `./build/group_gemm_bench --help`。默认预热 20 次，每个计时批次 100 次操作，正式测量 7 轮。`--iterations`、`--warmup`、`--repeats` 可以修改这些数量。

## 两个场景与存储布局

数学运算是 `Y_e = A_e × W_eᵀ`，`alpha=1, beta=0`。A、W、Y 均按行存储，W 的物理形状为 `[4, N, K]`，每个专家使用自己的权重。

| 场景 | N | K | 有效行数 | 输入 A | 输出 Y |
|---|---:|---:|---|---|---|
| Prefill / contiguous | 4096 | 1024 | `[256,1024,128,512]` | `[1920,1024]` | `[1920,4096]` |
| Decode / masked | 4096 | 1024 | `[16,64,8,32]` | `[4,64,1024]` | `[4,64,4096]` |

contiguous 将各专家的 token 紧凑拼接，GPU 行偏移为 `[0,256,1280,1408,1920]`。空专家可以对应重复偏移。这个学习接口允许 M 不对齐，不强制 DeepGEMM 的专家分段对齐，也不声称兼容其 ABI。

masked 为每个专家预留 64 行，GPU `valid_rows` 表示各专家**前多少行有效**。无效输入行填为 NaN；无效输出行必须保持初始化值。本接口不支持任意位置的布尔 mask。

测试的是已经完成 token 分组后的单个专家线性层计算，不包含 MoE 路由、dispatch/combine、激活函数或网络通信。这里的四个专家可以看作本 GPU 上的本地专家。

本轮 CPU 预先知道各专家 M，设备上也有相同元数据。cuBLAS 的 M/N/K 数组在 host，因此这项实验不能证明它支持 GPU 动态行数的 masked 流程；相关同步/填充策略留给后续实验。

## 两个库做了什么

- `src/cublas_grouped_gemm.cu`：构造 GPU 地址表和 CPU 参数数组，`run()` 调用一次 `cublasGemmGroupedBatchedEx`。四个不同 M 对应 `group_count=4`、`group_size=[1,1,1,1]`；空专家在准备阶段排除。
- `src/cutlass_grouped_gemm.cu`：使用 `GemmGrouped` 和设备端调度器，`run()` 启动 grouped kernel。A 为 RowMajor，数学上的 B 为 ColumnMajor，因而直接使用物理行存储的 W，无需转换布局。
- `tests/test_correctness.cu`：数据生成、逐专家 FP32/CPU FP64 参考、有效输出与保护区域检查。
- `benchmarks/benchmarks.cu`：组织测试、配置筛选、预热、普通调用/Graph 计时与 CSV 输出。
- `include/grouped_gemm.h`：共同的借用视图、操作接口和资源管理；`include/custom_gemm.h` 定义自定义 kernel 的接口。

cuBLAS 使用 `CUBLAS_COMPUTE_32F`，并禁止低精度中间归约，预分配 32 MiB workspace。CUTLASS 使用 float accumulator 和 float epilogue compute，最后才转换为 FP16/BF16。两者的输入输出完全相同，分配、地址表准备和 workspace 初始化都不计时。

CUTLASS 候选均为 4 warp、3 stages，指令 tile 为 `16×8×16`：

| 配置 ID | Threadblock M×N×K | Warp M×N×K |
|---:|---|---|
| 0 | 32×128×32 | 32×32×32 |
| 1 | 64×128×32 | 32×64×32 |
| 2 | 128×128×32 | 64×64×32 |
| 3 | 128×128×64 | 64×64×64 |

每个场景和类型分别选择配置：先验证全部候选，再用 5 轮独立 Graph 测量选择中位数较低者。随后重新采集正式数据，普通调用和 Graph 使用同一配置。程序打印全部候选的测量结果，CSV 记录最终配置；这不代表遍历了 CUTLASS 的全部优化空间。

## 如何解释延迟和 TFLOPS

一次操作包含全部四个专家 GEMM。采用常见 GEMM 计数约定，一次乘加算 2 FLOPs：

```text
FLOPs  = 2 × N × K × sum(valid_M)
TFLOPS = FLOPs / (latency_us × 1,000,000)

Prefill: sum(M)=1920，FLOPs=16,106,127,360
Decode:  sum(M)=120， FLOPs= 1,006,632,960
```

例如 Decode 如果测得 10 us，有效吞吐量就是 100.663296 TFLOPS。填充行和 tile 内部的无效计算不加入分子，不能通过扩大容量让有效吞吐量虚高。

两种模式均使用同一个非默认 stream 上的 CUDA Event：

- **normal**：连续从 CPU 提交 100 次完整操作，测总时间，再除以 100。GPU 时间线可能包含 CPU 来不及提交造成的间隙；它不是整个 CPU API 的墙钟耗时。
- **graph**：提前将 100 次完整操作串行捕获到一个 Graph，测一次回放总时间，再除以 100。库调用内部的所有 GPU 节点都在捕获范围内。捕获、实例化和第一次回放检查不计时。

正式结果是 7 轮“批次平均每次耗时”的中位数，**不是单次请求延迟的 P50**。两个库按固定随机顺序轮换测量。Graph 延迟和 TFLOPS 反映稳定重复计算的表现，不是完整推理请求的端到端指标。

默认重复使用同一批输入和输出，不清空 L2。四组 16-bit 权重共 32 MiB，在 4090 上可能大量复用缓存，因此不能将结果当作每次权重都从显存读取的冷缓存性能。不要在其他 GPU 任务运行时采集用来比较的结果；程序本身不会终止任何已有任务。

默认 CSV 有 16 行：2 场景 × 2 类型 × 2 库 × 2 计时方式。包含有效 FLOPs、`latency_us`、`tflops`、正确性、最大绝对误差、配置、GPU/库版本和测量参数。CSV 会覆盖指定路径，请为需要保留的实验选择不同文件名。

### 本机实测记录（2026-11-14）

GPU 0：RTX 4090；CUDA runtime/driver 13.4，cuBLAS 13.8.0，CUTLASS 4.5.0。测量前后检查，两张 GPU 均无其他计算负载。本次未锁定 GPU 时钟，使用上文的默认预热、批次与轮数。

以下是首次运行的 Graph 结果；原 CSV 已原样移至本机的 [results/results.csv](results/results.csv)。该文件被 Git 忽略，不随源码分发，后续使用默认文件名运行会覆盖它。

| 场景 | 类型 | cuBLAS us | cuBLAS TFLOPS | CUTLASS us | CUTLASS TFLOPS | CUTLASS 配置 ID |
|---|---|---:|---:|---:|---:|---:|
| Prefill | FP16 | 121.989 | 132.029 | 106.103 | 151.797 | 2 |
| Decode | FP16 | 17.480 | 57.589 | 15.043 | 66.919 | 1 |
| Prefill | BF16 | 122.552 | 131.422 | 106.052 | 151.871 | 2 |
| Decode | BF16 | 17.517 | 57.465 | 15.063 | 66.828 | 0 |

这是当前尺寸、输入复用策略和候选集下的一次测量。相近候选的排名可能随采样和时钟变化，不能推广成所有 Group GEMM 都由某个库更快。

## 正确性证据

输入由固定随机种子 2026 生成。A 取 `[-1,1]`，W 取 `[-1,1]/sqrt(K)`，先量化为待测的 16-bit 类型，然后由这些量化值建立参考。

- 主场景对所有有效输出进行比较。参考使用四次独立 FP32 `cublasGemmEx`，采用 `CUBLAS_COMPUTE_32F_PEDANTIC` 禁用 TF32；参考输出保留 FP32，不提前量化为 16-bit。
- 每个主场景专家的首尾、中心和 16 个固定随机位置，再用 CPU FP64 点积检查 FP32 参考，容差 `1e-5 + 1e-5 * abs(reference)`。
- 小尺寸边界案例用 CPU FP64 全量参考，包括 N=80、K=96、M=`[1,17,0,63]`、全部为空，以及 masked 的 `[64,0,1,63]`。M 可不对齐，N/K 保持正的 8 的倍数，但也能覆盖候选 tile 的 N/K 尾部。
- 有效输出采用 `abs(actual-reference) <= atol + rtol*abs(reference)`。FP16 的 atol/rtol 均为 `1e-3`；BF16 均为 `1e-2`。有效输出出现 NaN/Inf 直接失败。
- masked 无效行以及 Y 前后各 64 个元素的 guard 必须逐位不变。所有 CUTLASS 候选均检查，Graph 实际回放后也检查。

正确性失败或 CUDA API 报错会导致非零退出，不为失败操作发布性能数据。大矩阵的 CPU 检查是抽查，完整数值比较仍依赖独立 FP32 GEMM 参考；内存检查器用于补充非法访问证据。

本次验证完成：CTest 正确性通过；完整默认基准和 Graph 回放检查通过；CSV 的 16 种组合、FLOPs 和 TFLOPS 单位换算已核对。两种类型共完成 50 次边界算子检查，以及主场景全部候选检查。

内存验证还实际执行了两种计时路径，命令如下，Compute Sanitizer 报告 **0 errors**：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  --log-file build/memcheck.log \
  ./build/group_gemm_bench --warmup 1 --iterations 2 --repeats 1 \
  --csv checks/memcheck_not_performance.csv
```

这个命令减少重复次数以控制检测成本；它产生的延迟受到插桩影响，不能用作性能结果。构建目录中保留 `benchmark.log`、`correctness.log`、`memcheck.log` 和 `memcheck_program.log`，构建产物不纳入 Git。

## 自定义 CUDA kernel

`src/cutsom_gemm.cu` 保留当前文件名，在 `group_gemm` 命名空间内提供以下入口：

```cpp
cudaError_t launch_contiguous(const ContiguousGemmArgs& args, cudaStream_t stream);
cudaError_t launch_masked(const MaskedGemmArgs& args, cudaStream_t stream);
```

两个外层 kernel 负责专家布局与 persistent tile 调度，共同调用 `__device__ __forceinline__ tile_gemm()`。这个函数复用当前 CTA 的线程、共享内存和 barrier，不产生子 kernel；整个 block 必须一起调用它。核心完成协作加载、FP32 累加和边界写回。

迁移后，`cutsom_gemm.cu` 中的 `tile_gemm()` 仅转发到 [tile_gemm_v0()](src/tilegemm/tilegemm_v0.cu)。原有计算代码与注释完整保留在 v0 中，两个布局的索引和调度仍留在原文件。

`tilegemm_v0.cu` 保存设备函数模板，通过 `#include "tilegemm/tilegemm_v0.cu"` 引入同一个 CUDA 翻译单元，保留编译期实例化与内联；不要把它额外添加为 CMake 的独立编译源。以后实现同样签名的 `tile_gemm_v1()`，包含 `tilegemm_v1.cu`，再把 `tile_gemm()` 内唯一的转发调用从 v0 改为 v1 即可。版本选择在编译期完成。

| 布局 | X/Y 的专家起始行 | 有效 M |
|---|---|---|
| contiguous | `row_offsets[e]` | `row_offsets[e+1] - row_offsets[e]` |
| masked | `e * capacity` | `valid_rows[e]` |

W 在两种布局中都从 `e * N * K` 个元素处开始。masked 仅计算有效行，既不读取无效 X 行，也不修改无效 Y 行。当前固定使用 tile `32×64×32`、128 threads，grid 的 CTA 数量取当前 GPU 的 SM 数；这是一组基础配置，尚未自动调优，也未使用 Tensor Core。

接口中的所有矩阵指针和元数据指针都在 GPU。维度、类型和 stream 由 host 传入。要求 N/K 为正的 8 的倍数，支持任意非负有效 M，FP32 累加；masked 只写有效行。空集合应合法返回。具体参数含义见头文件注释。

入口只提交计算并返回启动状态，不做分配、H2D/D2H 或同步；使用传入的 stream，并允许 CUDA Graph 捕获。所有元数据与矩阵在操作和 Graph 执行完成之前都保持有效。当前 benchmark 的尺寸固定，但你的 kernel 应从 GPU 元数据读取专家边界或有效行数。

```bash
cmake -S . -B build -DENABLE_CUSTOM=ON
cmake --build build -j 2
./build/group_gemm_bench --provider custom --check-only
./build/group_gemm_bench --csv with_custom.csv
```

小规模专用回归测试还覆盖零专家、M/N/K 尾部、跨专家 persistent 复用，并在不重新捕获 Graph 的情况下缩小 masked 的有效行数：

```bash
ctest --test-dir build -R '^custom_gemm_correctness$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/custom_gemm_test
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/custom_gemm_test
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/custom_gemm_test
```

默认 `ENABLE_CUSTOM=OFF`，打印 `custom: SKIP`。开启后 `--provider all` 会把自定义实现加入相同的正确性、两种计时和 CSV 流程。

实现参考：[cuBLAS Grouped API](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex)、[CUTLASS grouped 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped)、[DeepGEMM 接口思路](https://github.com/deepseek-ai/DeepGEMM#interfaces)。
