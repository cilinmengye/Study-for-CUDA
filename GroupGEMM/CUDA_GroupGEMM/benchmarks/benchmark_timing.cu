/*
 * benchmark_timing.cu
 * 预热、统计与多轮计时；沿用原来的测量顺序和中位数计算。
 */
#include "benchmark_timing.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace group_gemm {

void warm_up(GemmOperation& operation, cudaStream_t stream, int count) {
    // PERF: 排除首次加载、首次调用等一次性影响，并让输入有机会进入缓存。
    //       预热不计时；这也意味着后面的结果是重复使用输入时的性能，不是冷缓存结果。
    for (int i = 0; i < count; ++i) {
        operation.run();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}


double median(std::vector<double> samples) {
    // 这里的每个 sample 已经是“整批时间 / iterations”，因此取中位数
    // 得到的是批次平均值的中位数，不是独立单次请求延迟的 P50。
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2;
}


std::vector<std::vector<double>> measure_rounds(
    std::vector<std::unique_ptr<BatchTimer>>& timers, int rounds
) {
    // samples[i][r] 保存第 i 个后端/候选在第 r 轮的每次平均耗时，单位 us。
    std::vector<std::vector<double>> samples(timers.size());
    std::vector<int> order(timers.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(17);
    for (int round = 0; round < rounds; ++round) {
        // 轮换后端/候选的测量顺序，减小温度与时钟缓慢变化造成的顺序偏差。
        std::shuffle(order.begin(), order.end(), rng);
        for (int index : order) {
            samples[index].push_back(timers[index]->measure_us());
        }
    }
    return samples;
}

}  // namespace group_gemm
