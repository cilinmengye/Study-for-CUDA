/*
 * cublas_grouped_gemm.cu
 * 原生 cuBLAS Grouped GEMM 基线。准备阶段建立 host 尺寸数组和 device 地址表，
 * run() 每次只调用一次 cublasGemmGroupedBatchedEx。
 * 学习重点：两层地址表、row-major 到 column-major 的等价解释、group 的 API 含义。
 */
#include "grouped_gemm.h"

#include <algorithm>

namespace group_gemm {
namespace {

/*
 * 将 contiguous / masked 的专家切片适配到 cuBLAS 的列存储接口。
 * problem: [Host] 描述结构；其中 A/W/Y 指针均为 [Device]，rows 为 [Host]。
 * 本类拥有地址表、工作空间和 handle，但不拥有 problem 中的矩阵内存。
 */
class CublasGrouped final : public GemmOperation {
public:
    CublasGrouped(const GroupedProblem& problem, cudaStream_t stream)
        : type_(problem.dtype == DType::FP16 ? CUDA_R_16F : CUDA_R_16BF),
          count_(std::count_if(problem.rows.begin(), problem.rows.end(), [](int m) { return m > 0; })),
          // 注意类中的重要变量在此进行初始化:
          w_ptrs_(count_ * sizeof(void*)),
          a_ptrs_(count_ * sizeof(void*)),
          y_ptrs_(count_ * sizeof(void*)),
          workspace_(32 * 1024 * 1024),
          handle_(stream) {
        // NOTE: CPU 上的尺寸提前已知；这里没有从 GPU valid_rows 回读数据。
        //       w_addresses 是 CPU vector，但 vector 中的每个值都是 GPU 上某个 W_e 的地址。
        //       上传后形成两层访问：device 地址表 -> device 矩阵，不是 host 地址表直接给 kernel。
        std::vector<const void*> w_addresses, a_addresses;
        std::vector<void*> y_addresses;
        for (int e = 0; e < problem.experts(); ++e) {
            if (problem.rows[e] == 0) {
                // 空专家不生成库任务，输出仍保留原值。
                continue;
            }

            const auto row = problem.row_offset(e);
            // NOTE: 先从总缓冲区定位专家切片。W 每组固定 N*K 个元素，A/Y 起点由布局决定。
            //       uint16_t* 只用于让指针每加 1 前进 2 字节；没有把 FP16/BF16 数值转为整数。
            //       int64_t(e) 保证在计算大型张量的元素偏移时不会先发生 32-bit 溢出。
            w_addresses.push_back(static_cast<const uint16_t*>(problem.w) + int64_t(e) * problem.n * problem.k);
            a_addresses.push_back(static_cast<const uint16_t*>(problem.a) + row * problem.k);
            y_addresses.push_back(static_cast<uint16_t*>(problem.y) + row * problem.n);

            // NOTE: 先看原始数据：总权重是行主序 W_all[E,N,K]，当前专家取其中的 W[N,K]。
            //       当前专家的输入 A[M,K]、输出 Y[M,N] 也都是行主序，M = problem.rows[e]。
            //       我们要算的是 Y = A * W^T；每个 token 的 K 维输入变成 N 维输出。
            //
            // cuBLAS 按列主序读写矩阵；指针本身不带 shape，形状由尺寸和转置参数指定。
            // 我们把 W 的同一块内存描述为列主序 [K,N]，记作 W_view，它恰好等于 W^T。
            // 例如 N=2、K=3（只示意存储顺序）：
            //   原始 W[2,3]：  [1 2 3]    显存顺序：[1,2,3,4,5,6]
            //                  [4 5 6]
            //   列主序读取为   [1 4]      第一列取 1,2,3，第二列取 4,5,6。
            //   W_view[3,2]：  [2 5]
            //                  [3 6]
            // 同理，A 的列主序视图是 A_view[K,M] = A^T，Y 的是 Y_view[N,M] = Y^T。

            // NOTE: 为使 cuBLAS 写出的列主序 Y_view 正好是所需的行主序 Y，计算 Y^T = W * A^T。
            //       因此第一个操作数传 w，第二个传 a；它们的处理分别是：
            //         W_view[K,N] --OP_T--> W[N,K]    （把上例的 3×2 视图转回 2×3）
            //         A_view[K,M] --OP_N--> A^T[K,M]  （已是所需的 A^T，无需再转置）
            //       OP_T 指定乘法使用视图的转置；这里没有另行分配、拷贝一份转置矩阵。
            trans_w_.push_back(CUBLAS_OP_T);
            trans_a_.push_back(CUBLAS_OP_N);

            // NOTE: cuBLAS 的 (m,n,k) 描述 op(第一个矩阵)[m,k] * op(第二个矩阵)[k,n]。
            //       此处是 [N,K] * [K,M] -> [N,M]，所以 (m,n,k)=(N,M,K)。
            //       业务中的 token 行数 M 对应下面的 n_，输出维度 N 对应 m_。
            m_.push_back(problem.n);
            n_.push_back(problem.rows[e]);
            k_.push_back(problem.k);

            // NOTE: leading dimension 是存储视图中相邻两列起点的元素间距，按应用 OP_T 之前算。
            //       W_view[K,N]、A_view[K,M] 的列间距都是 K，所以 run() 用 k_ 同时传 lda/ldb。
            //       Y_view[N,M] 的列间距是 N，所以 ldc=N。
            ldc_.push_back(problem.n);

            // NOTE: 本示例一位专家对应一个 cuBLAS group，group 内仅有一次 GEMM。
            //       group_count=count_，group_size=[1,...,1]；总 GEMM 数是 sum(group_size)。
            //       即 同一个 group 内的 GEMM 必须规格相同，不同 group 之间可以不同。
            //       因为我们并不确定输入样例，这里默认每个专家处理的 M 不同。
            groups_.push_back(1);
            // beta=0 表示 Y = alpha*A*B，不累加旧 Y；重复基准调用不会使结果不断变大。
            alpha_.push_back(1.0f);
            beta_.push_back(0.0f);
        }

        // 这里只拷贝地址值（每项 sizeof(void*) 字节），矩阵内容已经由 Dataset 上传。
        // 尺寸、转置、alpha/beta、group_size 等数组仍保留在 CPU，遵循 cuBLAS 的接口约定。
        w_ptrs_.upload(w_addresses.data(), stream);
        a_ptrs_.upload(a_addresses.data(), stream);
        y_ptrs_.upload(y_addresses.data(), stream);

        // PERF: workspace 在捕获/计时之前分配；不会把内部分配混入 Graph 回放。
        CUBLAS_CHECK(cublasSetWorkspace(handle_.get(), workspace_.get(), workspace_.bytes()));
        // NOTE: 输入输出用 16-bit 与“中间归约也用 16-bit”是两回事。
        //       这里禁止低精度中间归约，run() 再指定 FP32 compute，统一两库的计算口径。
        CUBLAS_CHECK(cublasSetMathMode(handle_.get(), CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    }


    void run() override {
        if (count_ == 0) {
            // 全空集合是合法的无操作，不调用库也不触碰输出。
            return;
        }

        // NOTE: 这是一次 grouped API 调用，不是在 CPU 上循环调用 count_ 次普通 GEMM。
        //       一次库调用内部可能包含多个 GPU kernel，benchmark 会统计整个调用的 GPU 工作。
        //       两处 k_.data() 分别充当 lda/ldb，因为上面的列存储视图中列间距都是 K。
        
        // API 通过 <cublas_v2.h> 引入。
        CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
            handle_.get(), trans_w_.data(), trans_a_.data(), m_.data(), n_.data(), k_.data(),
            alpha_.data(), w_ptrs_.get<const void*>(), type_, k_.data(),
            a_ptrs_.get<const void*>(), type_, k_.data(), beta_.data(),
            y_ptrs_.get<void*>(), type_, ldc_.data(), count_, groups_.data(), CUBLAS_COMPUTE_32F
        ));
    }


    std::string config() const override { return "native grouped; FP32 accumulation; workspace=32MiB"; }

private:
    cudaDataType_t type_;
    int count_;
    // [Host] 每个 group 的运算描述；保留为成员以覆盖重复调用与 Graph 捕获的生命周期。
    std::vector<cublasOperation_t> trans_w_, trans_a_;
    std::vector<int> m_, n_, k_, ldc_, groups_;
    std::vector<float> alpha_, beta_;
    // [Device] 前三个缓冲区保存矩阵地址，workspace 保存库使用的临时数据。
    DeviceBuffer w_ptrs_, a_ptrs_, y_ptrs_, workspace_;
    BlasHandle handle_;
};

}  // namespace


OperationPtr make_cublas(const GroupedProblem& problem, cudaStream_t stream) {
    return std::make_unique<CublasGrouped>(problem, stream);
}

}  // namespace group_gemm
