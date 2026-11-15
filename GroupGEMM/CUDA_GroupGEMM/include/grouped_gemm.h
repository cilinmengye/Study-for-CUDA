/*
 * grouped_gemm.h
 * 两个库共用的输入约定、操作接口和少量 CUDA 资源管理工具。
 * 所有矩阵均为连续行存储；数学运算是 Y_e = A_e * W_e^T。
 * 阅读顺序：资源生命周期 -> GroupedProblem 的内存布局 -> GemmOperation 的调用约定。
 * NOTE: 本文件只描述输入与 host 包装层，真正的 GPU 矩阵乘由两个库实现。
 */
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace group_gemm {



// ===========================
// 错误检查与资源生命周期
// ===========================

inline void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        // 保留原始调用名，便于定位异步 CUDA 故障。
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    }
}


inline void check_cublas(cublasStatus_t status, const char* expression) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        // cuBLAS 状态码与 CUDA runtime 状态码不能混用。
        throw std::runtime_error(std::string(expression) + ": cuBLAS status " + std::to_string(status));
    }
}

// NOTE: #call 将调用表达式转为错误信息中的字符串；(call) 本身只执行一次。
//       kernel 启动通常异步返回，部分设备错误要到后续同步点才会被观察到。
#define CUDA_CHECK(call) ::group_gemm::check_cuda((call), #call)
#define CUBLAS_CHECK(call) ::group_gemm::check_cublas((call), #call)


/*
 * 独占一段设备内存；只在准备阶段分配，禁止隐式复制所有权。
 * 这是 RAII：构造函数获取资源，析构函数释放资源，异常退出也会走析构。
 * bytes 以字节为单位，例如存放 count 个 FP16/BF16 时需要 count * 2 字节。
 * NOTE: 若允许直接复制对象，两个对象会保存同一个指针并重复 cudaFree，所以禁用复制。
 */
class DeviceBuffer {
public:
    // explicit 显式构造，否则使用例如 DeviceBuffer buf = 1024; 的方式，编译器
    // 也会构造出 DeviceBuffer, 调用如下函数进行 cudaMalloc
    explicit DeviceBuffer(size_t bytes = 0) : bytes_(bytes) {
        if (bytes != 0) {
            // 空专家集合不需要分配零字节内存。
            CUDA_CHECK(cudaMalloc(&pointer_, bytes));
        }
    }


    ~DeviceBuffer() { cudaFree(pointer_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;


    /* 返回 [Device] 指针；类型转换不拷贝数据，只决定指针运算的元素步长。
     * WARNING: CPU 可以保存该地址、计算偏移，但不能直接用 *get<T>() 读取显存。
     */
    template<class T = void>
    T* get() const { return static_cast<T*>(pointer_); }


    size_t bytes() const { return bytes_; }


    /* source: [Host]，至少包含 bytes() 字节；stream: 调用者的 CUDA stream。
     * NOTE: 同步到 stream 完成，保证局部 host vector 可在此函数返回后销毁。
     *       只用于计时之外的准备工作。
     */
    void upload(const void* source, cudaStream_t stream) {
        if (bytes_ != 0) {
            // 保证 H2D 与之前同一 stream 的操作按顺序执行。
            CUDA_CHECK(cudaMemcpyAsync(pointer_, source, bytes_, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

private:
    void* pointer_ = nullptr;
    size_t bytes_ = 0;
};


/*
 * 独占 cuBLAS handle；构造时绑定 stream，避免在每次计时调用中重复设置。
 * handle 是 host 侧管理对象，记录 stream、数学模式等；它本身不是矩阵数据。
 * 后续库调用会把 GPU 工作提交到这个 stream，与同 stream 上的 CUDA Event 保持顺序。
 */
class BlasHandle {
public:
    explicit BlasHandle(cudaStream_t stream) {
        CUBLAS_CHECK(cublasCreate(&handle_));
        const auto status = cublasSetStream(handle_, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            // 构造失败时析构函数不会运行，因此先释放已获得的 handle。
            cublasDestroy(handle_);
            CUBLAS_CHECK(status);
        }
    }


    ~BlasHandle() { cublasDestroy(handle_); }

    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;


    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};




// ===========================
// 数据类型与专家的物理布局
// ===========================

enum class DType { FP16, BF16 };
enum class Layout { Contiguous, Masked };


inline const char* dtype_name(DType type) { return type == DType::FP16 ? "fp16" : "bf16"; }


inline const char* layout_name(Layout layout) {
    return layout == Layout::Contiguous ? "contiguous" : "masked";
}


/*
 * 库适配器的借用视图，不拥有矩阵内存。
 * a, w, y: [Device] 16-bit 元素；w 的物理形状是 [experts, n, k]。
 * metadata: [Device] int32；contiguous 为 experts+1 个行偏移，masked 为有效行数。
 * rows: [Host] 本轮固定的有效行数，供库的准备阶段使用。
 * capacity: masked 中每个专家预留的物理行数；contiguous 不使用此字段。
 * 这里的“视图”只保存地址和尺寸；复制视图不会复制 A/W/Y 的实际内容。
 * rows 是 host vector，metadata 则是对应的 GPU 元数据，两者不是同一块内存。
 * NOTE: 输入、输出不重叠，生命周期覆盖操作、Graph 及其最后一次 GPU 执行。
 */
struct GroupedProblem {
    DType dtype;
    Layout layout;
    int n;
    int k;
    int capacity;
    std::vector<int> rows;
    // `a/w/y` 分别指向汇集了多个 expert 数据的 GPU 缓冲区。
    const void* a;
    const void* w;
    void* y;
    const int32_t* metadata;


    int experts() const { return static_cast<int>(rows.size()); }


    int64_t row_offset(int expert) const {
        // NOTE: 返回的是“行”的偏移，不是字节或元素偏移。
        //       A 的元素偏移 = row_offset * K，Y 的元素偏移 = row_offset * N。
        //       例如第 3 号专家：Prefill 起点是 256+1024+128=1408 行；
        //       Decode 起点是 3*64=192 行，不能用有效行数 16+64+8 代替。
        if (layout == Layout::Masked) {
            // 固定容量布局中，专家起点与有效行数无关。
            return int64_t(expert) * capacity;
        }
        // 返回 向量 rows 从下标0开始，到下标 expert - 1 这段区间 [0, expert) 值的累积
        return std::accumulate(rows.begin(), rows.begin() + expert, int64_t{0});
    }


    int64_t flops() const {
        // NOTE: 每个有效 Y 元素做 K 次乘加，按常用约定每次乘加算 2 FLOPs。
        //       只累加有效 M，masked padding 和 tile 内部的冗余计算不进入分子。
        //       2LL 从第一次乘法就使用 64-bit；Prefill 的约 161 亿 FLOPs 超过 int32。
        return 2LL * n * k * std::accumulate(rows.begin(), rows.end(), int64_t{0});
    }
};




// ===========================
// 准备阶段与重复执行阶段的分界
// ===========================

/*
 * 初始化后的可重复操作；run() 只提交计算，不分配、不拷贝、不主动同步。
 * make_cublas / make_cutlass 负责创建对象并准备元数据，不放入性能计时区间。
 * benchmark 通过相同的 run() 接口调用不同库，从而使用相同的预热与计时流程。
 * WARNING: run() 返回只表示已提交，不能据此释放 A/W/Y；必须等 stream 中工作完成。
 *          Graph 也借用这些地址，因此 Graph 执行结束前对象与数据都必须存活。
 */
class GemmOperation {
public:
    virtual ~GemmOperation() = default;
    virtual void run() = 0;
    virtual std::string config() const = 0;
};

using OperationPtr = std::unique_ptr<GemmOperation>;

OperationPtr make_cublas(const GroupedProblem& problem, cudaStream_t stream);
OperationPtr make_cutlass(const GroupedProblem& problem, int config, cudaStream_t stream);
std::string cutlass_version();
constexpr int kCutlassConfigs = 4;

}  // namespace group_gemm
