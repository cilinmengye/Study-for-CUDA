/*
 * custom_operation.h
 * 自定义 Group GEMM 的 benchmark 适配器；仅负责保存配置和转发公共启动接口。
 */
#pragma once

#include "custom_gemm.h"

#include <utility>

namespace group_gemm {

// ===========================
// 用户接口适配与候选验证
// ===========================

#ifdef ENABLE_CUSTOM
/* 只做参数适配，计算函数由 src/cutsom_gemm.cu 提供。 */
class CustomOperation final : public GemmOperation {
public:
    CustomOperation(
        GroupedProblem problem,
        cudaStream_t stream,
        int config = 0
    )
        : problem_(std::move(problem)),
          stream_(stream),
          config_(config) {}


    void run() override {
        if (problem_.layout == Layout::Contiguous) {
            // GPU 偏移包含专家边界；不把 CPU rows 传给用户 kernel。
            const ContiguousGemmArgs args{
                problem_.a, problem_.w, problem_.y, problem_.metadata,
                problem_.experts(), problem_.n, problem_.k, problem_.dtype
            };
            CUDA_CHECK(launch_contiguous(args, stream_, config_));
        } else {
            // 固定容量和 GPU 有效行数共同定义本轮 masked 工作量。
            const MaskedGemmArgs args{
                problem_.a, problem_.w, problem_.y, problem_.metadata,
                problem_.experts(), problem_.capacity,
                problem_.n, problem_.k, problem_.dtype
            };
            CUDA_CHECK(launch_masked(args, stream_, config_));
        }
    }


    std::string config() const override {
        return "custom_config=" + std::to_string(config_);
    }

private:
    GroupedProblem problem_;
    cudaStream_t stream_;
    int config_;
};
#endif

}  // namespace group_gemm
