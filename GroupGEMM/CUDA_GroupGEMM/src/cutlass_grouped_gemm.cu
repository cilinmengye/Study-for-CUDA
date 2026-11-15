/*
 * cutlass_grouped_gemm.cu
 * CUTLASS C++ Grouped GEMM 基线，使用兼容 RTX 4090 的 SM80 Tensor Core 路径。
 * 四个候选只改变 tile 形状，配置选择由 benchmark 在正式计时之外完成。
 * 学习重点：模板定义 GPU 算法，Arguments 描述本次输入，initialize() 准备，run() 执行。
 * 这些模板在 nvcc 构建时实例化；基准中选择配置并不会重新 JIT 编译 kernel。
 */
#include "grouped_gemm.h"

#include <cutlass/cutlass.h>
#include <cutlass/version.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#include <algorithm>

namespace group_gemm {
namespace {

void check_cutlass(cutlass::Status status) {
    if (status != cutlass::Status::kSuccess) {
        // 不静默退回其他配置，失败的配置不能参与排名。
        throw std::runtime_error(std::string("CUTLASS: ") + cutlassGetStatusString(status));
    }
}


/*
 * 同一设备端调度器同时处理两种布局：差异只在专家切片的起点。
 * problem: [Host] 尺寸/布局描述，其中矩阵地址均为 [Device]。
 * Element 是 FP16/BF16 类型，Tile 是 CTA 级分块，Warp 是每个 warp 的分块。
 * NOTE: Tile::kK 表示一次 K 循环处理多少列，不是输入矩阵的完整 K。
 */
template<class Element, class Tile, class Warp>
class CutlassGrouped final : public GemmOperation {
    // NOTE: 下面模板参数可以分成五组理解：
    // 1. A/B 的元素类型、布局、共轭变换与访问对齐。这里的 8 是“8 个元素”，即 16 字节。
    //    A[M,K] 是 RowMajor；B[K,N] 是 ColumnMajor，其字节排列恰好等于行存储 W[N,K]。
    // 2. 输出 Element + RowMajor，内部 accumulator=float，因此只在最终输出时舍入为 16-bit。
    // 3. OpClassTensorOp 选择 Tensor Core，Sm80 选择可用于本机 SM89 的 Ampere 指令实现路径。
    // 4. Tile/Warp/16x8x16 分别描述 CTA、warp 和底层 MMA 指令三个层次的矩阵形状。
    // 5. LinearCombination 实现 D=alpha*accumulator+beta*C；其中 8 是每次输出访问的元素数。
    //
    // 末尾的 3 指多 stage 流水线配置，用于组织 K 分块的数据搬运与计算重叠，
    // 不是 3 个 warp、3 个 kernel 或保证同时有 3 轮计算执行。
    // kDeviceOnly 表示 GPU 的 problem visitor 查找当前 tile 属于哪个专家；
    // “DeviceOnly”说的是调度方式，CPU 仍要在准备阶段建立问题尺寸和地址表。

    // using：这是 C++11 引入的关键字，用来定义类型别名。它的功能和老式的 typedef 类似，但可读性更好（特别是在处理模板时）。
    // typename 在这里的作用是告诉编译器：“这是一个类型（Type），而不是一个静态变量或函数。”
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
        Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
        Element, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        Tile, Warp, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
    >::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
    // 当前候选的 Tile::kK 与 Warp::kK 相同，warp 数由 M/N 两维的切分决定。
    // 例如 Tile=128x128x32、Warp=64x64x32，对应 (128/64)*(128/64)=4 个 warp。
    static_assert(Kernel::kThreadCount == 128, "Every candidate must use four warps");

public:
    CutlassGrouped(const GroupedProblem& problem, cudaStream_t stream)
        : stream_(stream),
          count_(std::count_if(problem.rows.begin(), problem.rows.end(), [](int m) { return m > 0; })),
          sizes_device_(count_ * sizeof(cutlass::gemm::GemmCoord)),
          // 输入精度	Element 的具体类型	Element* 的具体类型
          // FP16	cutlass::half_t	cutlass::half_t*
          // BF16	cutlass::bfloat16_t	cutlass::bfloat16_t*
          a_ptrs_(count_ * sizeof(Element*)),
          w_ptrs_(count_ * sizeof(Element*)),
          y_ptrs_(count_ * sizeof(Element*)),
          ld_input_(count_ * sizeof(int64_t)),
          ld_output_(count_ * sizeof(int64_t)) {
        if (count_ == 0) {
            // 空集合不查询 occupancy，也不初始化零规模调度器。
            return;
        }

        // CUTLASS 与 cuBLAS 不同：本实现的 problem_sizes、地址表、leading dimensions
        // 都会上传 GPU。host vectors 只是用于构造这些 device 数组。
        std::vector<Element*> a_addresses, w_addresses, y_addresses;
        std::vector<int64_t> ld_input(count_, problem.k), ld_output(count_, problem.n);
        for (int e = 0; e < problem.experts(); ++e) {
            if (problem.rows[e] == 0) {
                // 保留物理布局中的专家位置，仅从任务列表中移除空专家。
                continue;
            }

            // 这里直接描述数学上的 A[M,K]*B[K,N]，不需要像 cuBLAS 适配时交换 M/N。
            sizes_.emplace_back(problem.rows[e], problem.n, problem.k);
            const auto row = problem.row_offset(e);
            // NOTE: CUTLASS 的地址表类型未保留 const，但 kernel 不写 A/W。
            // 三个向量分别保存各 expert 的输入、权重和输出的 GPU 起始地址。
            a_addresses.push_back(static_cast<Element*>(const_cast<void*>(problem.a)) + row * problem.k);
            w_addresses.push_back(static_cast<Element*>(const_cast<void*>(problem.w)) + int64_t(e) * problem.n * problem.k);
            y_addresses.push_back(static_cast<Element*>(problem.y) + row * problem.n);
        }

        sizes_device_.upload(sizes_.data(), stream);
        // 区分 a_addresses 和 a_ptrs_，是为了在 CPU 上构造地址表，再把地址表放到 GPU 上供 kernel 使用。
        a_ptrs_.upload(a_addresses.data(), stream);
        w_ptrs_.upload(w_addresses.data(), stream);
        y_ptrs_.upload(y_addresses.data(), stream);
        ld_input_.upload(ld_input.data(), stream);
        ld_output_.upload(ld_output.data(), stream);

        // ============= 将数据转移到 GEMM 相关 Arguments 类中 =================
        // NOTE: 行存储 W[N,K] 等价于列存储的 B[K,N]，无需转置拷贝。
        //       FP32 accumulator 与 epilogue compute，最后才转换为 FP16/BF16。
        typename Gemm::Arguments args;
        args.problem_sizes = sizes_device_.get<cutlass::gemm::GemmCoord>();
        args.problem_count = count_;
        // PERF: Grouped GEMM 通常使用 persistent 调度：一个 CTA 完成一个输出 tile 后，
        //       还可以继续处理其他 tile，甚至切换到另一个专家。
        //       sufficient() 根据 SM 数、kernel occupancy 和总 tile 数选择 CTA 数，
        //       所以 grid 大小不必等于专家数，也不必等于所有输出 tile 的总数。
        args.threadblock_count = Gemm::sufficient(sizes_.data(), count_);
        args.output_op = {1.0f, 0.0f};
        args.ptr_A = a_ptrs_.get<Element*>();
        args.ptr_B = w_ptrs_.get<Element*>();
        // NOTE: CUTLASS 将原有输出称为 C，最终写出称为 D；本项目二者都指向 Y。
        //       beta=0 时不需要旧 C 的数值，因此可重复覆盖同一个预分配输出。
        args.ptr_C = y_ptrs_.get<Element*>();
        args.ptr_D = y_ptrs_.get<Element*>();
        // leading dimension 的单位是元素，不是字节：
        // A RowMajor 的行间距为 K；B ColumnMajor 的列间距也为 K；C/D RowMajor 为 N。
        args.lda = ld_input_.get<int64_t>();
        args.ldb = ld_input_.get<int64_t>();
        args.ldc = ld_output_.get<int64_t>();
        args.ldd = ld_output_.get<int64_t>();
        // [Host] 这份副本供 host 辅助逻辑使用；GPU kernel 读取的是上面的 problem_sizes。
        args.host_problem_sizes = sizes_.data();

        if (args.threadblock_count <= 0 || Gemm::get_workspace_size(args) != 0) {
            // 当前选择设备端调度，要求有效 CTA 数且不需要 host-precompute workspace。
            throw std::runtime_error("Invalid CUTLASS occupancy or unexpected scheduler workspace");
        }
        check_cutlass(Gemm::can_implement(args));
        // initialize 缓存 kernel 参数并设置所需的动态 shared-memory 属性。
        // 本配置使用 device 调度，无需 host-precompute workspace；真正计算发生在 run()。
        check_cutlass(gemm_.initialize(args, nullptr, stream));
    }


    void run() override {
        if (count_ != 0) {
            // PERF: 元数据与 shared-memory 属性已准备好，此处只提交 grouped kernel。
            check_cutlass(gemm_.run(stream_));
        }
    }


    std::string config() const override {
        return "tb=" + shape<Tile>() + " warp=" + shape<Warp>() + " warps=4 stages=3";
    }

private:
    template<class Shape>
    static std::string shape() {
        return std::to_string(Shape::kM) + "x" + std::to_string(Shape::kN) + "x" + std::to_string(Shape::kK);
    }

    cudaStream_t stream_;
    int count_;
    std::vector<cutlass::gemm::GemmCoord> sizes_;
    // [Device] sizes_device_ 存 (M,N,K)；三个 ptrs_ 存地址；两个 ld_ 存各组元素步长。
    DeviceBuffer sizes_device_, a_ptrs_, w_ptrs_, y_ptrs_, ld_input_, ld_output_;
    Gemm gemm_;
};


template<class Element>
OperationPtr make_typed(const GroupedProblem& problem, int config, cudaStream_t stream) {
    using namespace cutlass::gemm;
    // NOTE: 四个分支对应预先编译的四种 kernel 类型，动态 config 只负责选其中一个。
    //       小 M 可以少浪费 tile 内的行，大 tile 可能增加复用；最终速度由 benchmark 决定。
    switch (config) {
        case 0: return std::make_unique<CutlassGrouped<Element, GemmShape<32, 128, 32>, GemmShape<32, 32, 32>>>(problem, stream);
        case 1: return std::make_unique<CutlassGrouped<Element, GemmShape<64, 128, 32>, GemmShape<32, 64, 32>>>(problem, stream);
        case 2: return std::make_unique<CutlassGrouped<Element, GemmShape<128, 128, 32>, GemmShape<64, 64, 32>>>(problem, stream);
        case 3: return std::make_unique<CutlassGrouped<Element, GemmShape<128, 128, 64>, GemmShape<64, 64, 64>>>(problem, stream);
        default: throw std::runtime_error("CUTLASS config must be in [0,3]");
    }
}

}  // namespace


OperationPtr make_cutlass(const GroupedProblem& problem, int config, cudaStream_t stream) {
    if (problem.dtype == DType::FP16) {
        // 仅类型分派不同；两种精度使用相同候选集与布局约定。
        return make_typed<cutlass::half_t>(problem, config, stream);
    }
    return make_typed<cutlass::bfloat16_t>(problem, config, stream);
}


std::string cutlass_version() {
    return std::to_string(CUTLASS_MAJOR) + "." + std::to_string(CUTLASS_MINOR) + "." + std::to_string(CUTLASS_PATCH);
}

}  // namespace group_gemm
