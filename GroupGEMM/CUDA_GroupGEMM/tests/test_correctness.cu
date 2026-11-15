/*
 * test_correctness.cu
 * 数据准备、独立参考与输出检查。大矩阵用逐专家的严格 FP32 GEMM 作全量参考，
 * 再用 CPU FP64 抽查；小矩阵直接采用 CPU FP64 全量参考。
 * 阅读顺序：Dataset 准备输入 -> make_reference 建立预期值 -> validate/run/check_output。
 * 所有分配、传输、参考计算和验证均在性能计时区间之外。
 */
#include "testbed.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>

namespace group_gemm {
namespace {



// ===========================
// 低精度编码与独立 CPU 参考
// ===========================

uint16_t encode(float value, DType dtype) {
    // NOTE: 返回的是浮点数的 16-bit 位模式，不是把浮点数截断成整数。
    //       同样两个字节，FP16 和 BF16 的指数/尾数布局不同，必须按 dtype 编解码。
    uint16_t bits;
    if (dtype == DType::FP16) {
        // 在 host 上先量化，参考路径不能偷偷使用量化前的 FP32 输入。
        const __half half = __float2half_rn(value);
        std::memcpy(&bits, &half, sizeof(bits));
    } else {
        // BF16 与 FP16 都占两字节，但编码不同。
        const __nv_bfloat16 half = __float2bfloat16_rn(value);
        std::memcpy(&bits, &half, sizeof(bits));
    }
    return bits;
}


float decode(uint16_t bits, DType dtype) {
    if (dtype == DType::FP16) {
        // memcpy 避免通过不兼容类型解引用导致严格别名问题。
        __half half;
        std::memcpy(&half, &bits, sizeof(bits));
        return __half2float(half);
    }

    __nv_bfloat16 half;
    std::memcpy(&half, &bits, sizeof(bits));
    return __bfloat162float(half);
}


int64_t storage_rows(const CaseSpec& spec) {
    if (spec.rows.empty() || spec.n <= 0 || spec.k <= 0 || spec.n % 8 || spec.k % 8) {
        // 当前 Tensor Core 接口约定 N/K 是正的 8 的倍数。
        throw std::runtime_error("Require nonempty experts and positive N/K divisible by 8");
    }
    for (int m : spec.rows) {
        if (m < 0 || (spec.layout == Layout::Masked && (spec.capacity <= 0 || m > spec.capacity))) {
            // 有效行必须落在各专家的物理容量内。
            throw std::runtime_error("Invalid expert row count/capacity");
        }
    }
    return spec.layout == Layout::Masked
        ? int64_t(spec.rows.size()) * spec.capacity
        : std::accumulate(spec.rows.begin(), spec.rows.end(), int64_t{0});
}


double cpu_dot(const Dataset& dataset, int expert, int row, int column) {
    const auto problem = dataset.problem();
    const auto a_start = (problem.row_offset(expert) + row) * problem.k;
    const auto w_start = (int64_t(expert) * problem.n + column) * problem.k;
    // NOTE: Y_e[row,column] 是 A_e 的一行与 W_e 的一行点积。
    //       W 已按 [N,K] 存储，所以两行在 K 方向都连续；这里不生成 W^T 临时矩阵。
    //       用 double 乘加得到独立于 GPU 调度、Tensor Core 和地址表的数值参考。
    double sum = 0;
    for (int k = 0; k < problem.k; ++k) {
        sum += double(dataset.host_a[a_start + k]) * double(dataset.host_w[w_start + k]);
    }
    return sum;
}

}  // namespace



// ===========================
// 构造 GPU 数据集与专家元数据
// ===========================


Dataset::Dataset(CaseSpec case_spec, DType type, cudaStream_t cuda_stream)
    : spec(std::move(case_spec)),
      dtype(type),
      stream(cuda_stream),
      storage_rows(group_gemm::storage_rows(spec)),
      host_a(std::max(int64_t{1}, storage_rows * spec.k)),
      host_w(int64_t(spec.rows.size()) * spec.n * spec.k),
      a(host_a.size() * sizeof(uint16_t)),
      w(host_w.size() * sizeof(uint16_t)),
      y_with_guards((storage_rows * spec.n + 2 * kGuardElements) * sizeof(uint16_t)),
      metadata((spec.rows.size() + (spec.layout == Layout::Contiguous ? 1 : 0)) * sizeof(int32_t)),
      sentinel(encode(-123.0f, dtype)) {
    // NOTE: y_with_guards 的物理排列为 [前 guard | Y 数据区 | 后 guard]。
    //       每边 64 个 16-bit 元素，即 128 字节，使实际 Y 起点仍保持 16 字节对齐。
    //       -123 在 FP16/BF16 中都可精确表示；有效结果应覆盖它，无效区域必须保留它。
    //       全空 contiguous 输入仍分配一个占位元素，避免构造零大小输入时做空指针运算。
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<uint16_t> a_bits(host_a.size(), encode(std::numeric_limits<float>::quiet_NaN(), dtype));
    std::vector<uint16_t> w_bits(host_w.size());
    const auto view = problem();

    // NOTE: masked 的无效输入行填 NaN，比填 0 更容易发现错误的跨行读取。
    //       有效区域随后会覆盖为随机数；错误地让 padding 参与有效结果，往往会传播 NaN。
    for (int e = 0; e < view.experts(); ++e) {
        const auto start = view.row_offset(e) * spec.k;
        for (int64_t i = 0; i < int64_t(spec.rows[e]) * spec.k; ++i) {
            a_bits[start + i] = encode(distribution(rng), dtype);
        }
    }

    // 控制输出尺度，避免溢出或仅因数值过大而掩盖布局错误。
    // W 除以 sqrt(K)，使 K 变大时点积的典型尺度不会同步变得很大。
    // host_a/host_w 保存 decode(encode(x))：参考计算使用实际量化值，
    // 而不是精度更高、但 GPU 从来没有收到过的原始随机 float。
    const float scale = 1.0f / std::sqrt(float(spec.k));
    for (size_t i = 0; i < w_bits.size(); ++i) {
        w_bits[i] = encode(distribution(rng) * scale, dtype);
        host_w[i] = decode(w_bits[i], dtype);
    }
    for (size_t i = 0; i < a_bits.size(); ++i) {
        host_a[i] = decode(a_bits[i], dtype);
    }
    a.upload(a_bits.data(), stream);
    w.upload(w_bits.data(), stream);

    std::vector<int32_t> host_metadata;
    if (spec.layout == Layout::Contiguous) {
        // 前缀和同时表达专家边界和总行数；空专家产生重复偏移。
        // 例如 rows=[1,17,0,63] 得到 offsets=[0,1,18,18,81]。
        // 专家 2 行数为 offsets[3]-offsets[2]=0，专家 3 紧接着从第 18 行开始。
        host_metadata.push_back(0);
        for (int m : spec.rows) {
            host_metadata.push_back(host_metadata.back() + m);
        }
    } else {
        // masked 元数据只有有效行数，物理起点由 capacity 决定。
        // 相同 rows 在 capacity=64 时的物理起点始终为 [0,64,128,192]，不能紧凑拼接。
        host_metadata.assign(spec.rows.begin(), spec.rows.end());
    }
    metadata.upload(host_metadata.data(), stream);
    reset_output();
}


GroupedProblem Dataset::problem() const {
    // NOTE: y 指向数据区，而不是 guard；库无需知道测试额外分配的保护区域。
    //       返回对象只复制地址与描述，GPU 大矩阵本身不会被复制。
    return {
        dtype, spec.layout, spec.n, spec.k, spec.capacity, spec.rows,
        a.get(), w.get(), y_with_guards.get<uint16_t>() + kGuardElements, metadata.get<int32_t>()
    };
}


void Dataset::reset_output() {
    std::vector<uint16_t> initial(y_with_guards.bytes() / sizeof(uint16_t), sentinel);
    y_with_guards.upload(initial.data(), stream);
}


std::vector<uint16_t> Dataset::download_output() const {
    std::vector<uint16_t> output(y_with_guards.bytes() / sizeof(uint16_t));
    CUDA_CHECK(cudaMemcpyAsync(output.data(), y_with_guards.get(), y_with_guards.bytes(), cudaMemcpyDeviceToHost, stream));
    // WARNING: D2H 提交不等于 host 数据已经可读；必须等待同 stream 中计算和拷贝完成。
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return output;
}




// ===========================
// 全量参考与数值、边界检查
// ===========================

std::vector<float> make_reference(const Dataset& dataset, bool full_cpu) {
    const auto problem = dataset.problem();
    std::vector<float> reference(dataset.storage_rows * problem.n, 0.0f);
    if (full_cpu) {
        // 小矩阵的所有有效元素都由独立 CPU FP64 点积给出。
        // 最后转为 float 保存参考；比较时不会先将参考舍入到被测的 16-bit 输出精度。
        for (int e = 0; e < problem.experts(); ++e) {
            for (int m = 0; m < problem.rows[e]; ++m) {
                for (int n = 0; n < problem.n; ++n) {
                    reference[(problem.row_offset(e) + m) * problem.n + n] = float(cpu_dot(dataset, e, m, n));
                }
            }
        }
        return reference;
    }

    DeviceBuffer a_float(dataset.host_a.size() * sizeof(float));
    DeviceBuffer w_float(dataset.host_w.size() * sizeof(float));
    DeviceBuffer y_float(reference.size() * sizeof(float));
    a_float.upload(dataset.host_a.data(), dataset.stream);
    w_float.upload(dataset.host_w.data(), dataset.stream);
    y_float.upload(reference.data(), dataset.stream);
    BlasHandle handle(dataset.stream);
    const float alpha = 1.0f, beta = 0.0f;

    // NOTE: 这是参考路径，不是性能基线。逐专家调用普通 GEMM，不使用 grouped 地址表，
    //       输入/输出均为 FP32，PEDANTIC 禁止 TF32，结果不提前舍入为 16-bit。
    // 与 grouped 路径相比，这里不复用其 GPU 地址表，也不依赖 CUTLASS 的 problem visitor。
    // 但它仍然调用 cuBLAS，所以后面还用 CPU FP64 抽查，避免只让两个库互相比较。
    for (int e = 0; e < problem.experts(); ++e) {
        if (problem.rows[e] == 0) {
            // 没有有效输出的专家不参与参考计算。
            continue;
        }

        const auto row = problem.row_offset(e);
        CUBLAS_CHECK(cublasGemmEx(
            handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, problem.n, problem.rows[e], problem.k,
            &alpha, w_float.get<float>() + int64_t(e) * problem.n * problem.k, CUDA_R_32F, problem.k,
            a_float.get<float>() + row * problem.k, CUDA_R_32F, problem.k, &beta,
            y_float.get<float>() + row * problem.n, CUDA_R_32F, problem.n,
            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT
        ));
    }
    CUDA_CHECK(cudaMemcpyAsync(reference.data(), y_float.get(), y_float.bytes(), cudaMemcpyDeviceToHost, dataset.stream));
    CUDA_CHECK(cudaStreamSynchronize(dataset.stream));

    // 独立检查 FP32 库参考：包含专家首尾行、首尾列及固定随机位置。
    std::mt19937 rng(71);
    for (int e = 0; e < problem.experts(); ++e) {
        if (problem.rows[e] == 0) {
            // 避免对空专家生成无效的随机坐标。
            continue;
        }

        std::vector<std::pair<int, int>> positions = {
            {0, 0}, {0, problem.n - 1},
            {problem.rows[e] - 1, 0}, {problem.rows[e] - 1, problem.n - 1},
            {problem.rows[e] / 2, problem.n / 2}
        };
        for (int sample = 0; sample < 16; ++sample) {
            positions.emplace_back(rng() % problem.rows[e], rng() % problem.n);
        }
        for (auto [m, n] : positions) {
            const double expected = cpu_dot(dataset, e, m, n);
            const float actual = reference[(problem.row_offset(e) + m) * problem.n + n];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5 + 1e-5 * std::abs(expected)) {
                // 参考路径不可信时停止测试，不能用它给待测 kernel 背书。
                throw std::runtime_error("FP32 reference disagrees with independent CPU FP64 sample");
            }
        }
    }
    return reference;
}


CheckResult check_output(const Dataset& dataset, const std::vector<float>& reference) {
    const auto output = dataset.download_output();
    const auto problem = dataset.problem();
    const double atol = dataset.dtype == DType::FP16 ? 1e-3 : 1e-2;
    const double rtol = dataset.dtype == DType::FP16 ? 1e-3 : 1e-2;
    // NOTE: 绝对误差 atol 保护接近 0 的参考值，避免相对误差除以极小数；
    //       rtol 则让允许误差随结果大小变化。BF16 尾数精度低于 FP16，因此阈值不同。
    //       即使 FP32 累加完全一致，最终写回 16-bit 时也会发生舍入，不能要求逐位等于 FP32。
    CheckResult result;
    size_t numerical_failures = 0, modified_padding = 0;
    std::string first_failure;

    for (int e = 0; e < problem.experts(); ++e) {
        const auto start = problem.row_offset(e) * problem.n;
        for (int64_t i = 0; i < int64_t(problem.rows[e]) * problem.n; ++i) {
            // reference 的下标从数据区 0 开始；下载的 output 包括前 guard，所以额外加偏移。
            const float actual = decode(output[Dataset::kGuardElements + start + i], dataset.dtype);
            const float expected = reference[start + i];
            const double error = std::abs(double(actual) - expected);
            ++result.checked_elements;
            result.max_abs_error = std::max(result.max_abs_error, error);
            if (!std::isfinite(actual) || !std::isfinite(expected) || error > atol + rtol * std::abs(expected)) {
                // 有效行必须完整写入且有限，不能将 NaN 当作比较通过。
                // NOTE: NaN 与阈值做大小比较通常返回 false，只写 error > tolerance 会漏掉 NaN。
                ++numerical_failures;
                if (first_failure.empty()) {
                    // 只记录首个失败位置，避免错误输出淹没诊断信息。
                    std::ostringstream message;
                    message << " expert=" << e << " row=" << i / problem.n << " col=" << i % problem.n
                            << " actual=" << actual << " expected=" << expected;
                    first_failure = message.str();
                }
            }
        }

        if (problem.layout == Layout::Masked) {
            // 无效输出行不属于本次计算，要求逐位保持初始化值。
            for (int64_t i = int64_t(problem.rows[e]) * problem.n; i < int64_t(problem.capacity) * problem.n; ++i) {
                modified_padding += output[Dataset::kGuardElements + start + i] != dataset.sentinel;
            }
        }
    }

    const auto logical_elements = dataset.storage_rows * problem.n;
    // guard 只能发现写到保护区域的错误；它不能替代 Compute Sanitizer 对非法访问的检查。
    for (size_t i = 0; i < Dataset::kGuardElements; ++i) {
        modified_padding += output[i] != dataset.sentinel;
        modified_padding += output[Dataset::kGuardElements + logical_elements + i] != dataset.sentinel;
    }
    if (numerical_failures || modified_padding) {
        // padding/guard 被写坏同样判失败，即使所有有效元素的数值都正确。
        throw std::runtime_error(
            dataset.spec.name + "/" + dtype_name(dataset.dtype) + ": numerical_failures=" +
            std::to_string(numerical_failures) + " modified_padding_or_guard=" +
            std::to_string(modified_padding) + first_failure
        );
    }
    return result;
}


CheckResult validate(GemmOperation& operation, Dataset& dataset, const std::vector<float>& reference) {
    // NOTE: 每个候选先重置 Y，避免前一个正确 kernel 的结果掩盖当前 kernel 的漏写。
    //       库 run() 异步提交；check_output 内的 D2H + stream 同步确保读取的是本次结果。
    dataset.reset_output();
    operation.run();
    return check_output(dataset, reference);
}


std::vector<CaseSpec> primary_cases() {
    return {
        {"prefill", Layout::Contiguous, 4096, 1024, {256, 1024, 128, 512}, 0},
        {"decode", Layout::Masked, 4096, 1024, {16, 64, 8, 32}, 64}
    };
}


std::vector<CaseSpec> edge_cases() {
    // NOTE: N=80、K=96 都满足 8 元素对齐，但不是所有候选 tile 尺寸的整倍数。
    //       这样同时覆盖 N/K tile 尾部、M 非对齐、空专家、全空集合以及容量上边界。
    //       比如 BK=64 时 K=96 的最后一轮只有 32 个有效 K 元素。
    return {
        {"tail_empty", Layout::Contiguous, 80, 96, {1, 17, 0, 63}, 0},
        {"tail_empty", Layout::Masked, 80, 96, {1, 17, 0, 63}, 64},
        {"all_empty", Layout::Contiguous, 80, 96, {0, 0, 0, 0}, 0},
        {"all_empty", Layout::Masked, 80, 96, {0, 0, 0, 0}, 64},
        {"full_capacity", Layout::Masked, 80, 96, {64, 0, 1, 63}, 64}
    };
}

}  // namespace group_gemm
