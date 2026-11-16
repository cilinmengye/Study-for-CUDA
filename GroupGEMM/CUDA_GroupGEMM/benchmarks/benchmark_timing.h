/*
 * benchmark_timing.h
 * CUDA stream、Event 与 Graph 的资源管理；类内方法随类原样迁移，便于集中阅读。
 */
#pragma once

#include "grouped_gemm.h"

#include <nvtx3/nvToolsExt.h>

namespace group_gemm {

// ===========================
// CUDA stream、Graph 与批次计时
// ===========================

/* 独占一个非默认 stream，使所有后端和 event 使用同一执行队列。
 * NOTE: 同一个 stream 中 GPU 工作保持提交顺序；NonBlocking 表示不与 legacy default
 *       stream 建立隐式同步关系，不代表同 stream 中的 GEMM 可以越过前一个 GEMM 执行。
 */
class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }


    ~Stream() { cudaStreamDestroy(value); }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t value = nullptr;
};


/*
 * 不拥有 operation；Graph 和 event 必须先于 operation 及其输入销毁。
 * graph=false：每个测量批次由 CPU 循环提交 iterations 次 run()。
 * graph=true：在构造阶段捕获同样数量的 run()，每个批次只需提交一次 Graph 回放。
 * NOTE: 两种模式每批的数学工作量一致；区别在于 CPU 如何提交这些 GPU 工作。
 */
class BatchTimer {
public:
    BatchTimer(GemmOperation& operation, bool graph, int iterations, cudaStream_t stream)
        : operation_(operation), iterations_(iterations), stream_(stream) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
        if (graph) {
            // PERF: CUDA Graph, 捕获完整库调用序列，库内部即使有多个 kernel 也全部纳入计时。
            //       捕获时这段 for 将 GPU 操作记录成 Graph 节点；GPU 计算在回放时执行。
            //       所有节点引用已分配的同一份矩阵/元数据地址，Graph 不会复制矩阵内容。
            //       因为 beta=0 且不修改输入，同一 stream 串行覆盖 Y 是安全的。
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < iterations_; ++i) {
                operation_.run();
            }
            CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
            // cudaGraphInstantiateWithFlags() 得到可执行的 executable_
            CUDA_CHECK(cudaGraphInstantiateWithFlags(&executable_, graph_, 0));
        }
    }


    ~BatchTimer() {
        // NOTE: normal 模式没有 Graph；向 destroy 传空句柄会留下 CUDA 错误状态。
        if (executable_) cudaGraphExecDestroy(executable_);
        if (graph_) cudaGraphDestroy(graph_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    BatchTimer(const BatchTimer&) = delete;
    BatchTimer& operator=(const BatchTimer&) = delete;


    void launch_batch() {
        if (executable_) {
            // Graph 中已有 iterations 个串行操作，回放一次即可。
            // cudaGraphLaunch(executable_, stream_) 提交整张 Graph
            CUDA_CHECK(cudaGraphLaunch(executable_, stream_));
        } else {
            // 普通调用保留实际 host 提交方式，GPU 时间线可能出现提交间隙。
            for (int i = 0; i < iterations_; ++i) {
                // NOTE: 一次范围包含整个 Group GEMM 调用启动的所有 kernel。
                nvtxRangePushA("gemm");
                operation_.run();
                nvtxRangePop();
            }
        }
    }


    double measure_us() {
        // NOTE: Event 被排入同一条 GPU stream，start/stop 标记 GPU 时间线中的两个位置。
        //       这不是在 CPU 上用墙钟计时；但 normal 提交过慢时的 GPU 空等也会落在区间内。
        CUDA_CHECK(cudaEventRecord(start_, stream_));
        launch_batch();
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        // 等 stop 完成后才能读取时间差；等待本身在 CPU 上发生，不额外加入 event 的差值。
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        // CUDA 返回的是整批毫秒：先 *1000 转 us，再除以整组操作次数。
        // 例如整批 12.2 ms、100 次操作，对应每次 122 us，不能再除以专家数 4。
        return double(milliseconds) * 1000.0 / iterations_;
    }

private:
    GemmOperation& operation_;
    int iterations_;
    cudaStream_t stream_;
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};


void warm_up(GemmOperation& operation, cudaStream_t stream, int count);
double median(std::vector<double> samples);
std::vector<std::vector<double>> measure_rounds(
    std::vector<std::unique_ptr<BatchTimer>>& timers, int rounds
);

}  // namespace group_gemm
