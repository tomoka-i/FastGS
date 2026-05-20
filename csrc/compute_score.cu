#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 累積結果からスコアを計算するカーネル
__global__ void compute_final_score_kernel(
    const float* metric_score,
    const float* metric_counts,
    float* importance,
    float* pruning,
    int N,
    int num_views,
    float min_score,
    float score_range,
    bool densify)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Match the original Python logic:
    // pruning_score = (full_metric_score - min) / (max - min)
    // importance_score = floor(full_metric_counts / len(camlist))
    pruning[idx] = (metric_score[idx] - min_score) / score_range;
    importance[idx] = densify ? floorf(metric_counts[idx] / num_views) : 0.0f;
}

// C++ラッパー
std::tuple<torch::Tensor, torch::Tensor> compute_final_score_cuda(
    torch::Tensor metric_score,
    torch::Tensor metric_counts,
    int num_views,
    bool densify) 
{
    int N = metric_score.numel();
    auto importance = torch::zeros_like(metric_score);
    auto pruning = torch::zeros_like(metric_score);

    float min_score = metric_score.min().item<float>();
    float max_score = metric_score.max().item<float>();
    float score_range = max_score - min_score;

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    compute_final_score_kernel<<<grid, block>>>(
        metric_score.data_ptr<float>(),
        metric_counts.data_ptr<float>(),
        importance.data_ptr<float>(),
        pruning.data_ptr<float>(),
        N,
        num_views,
        min_score,
        score_range,
        densify
    );

    return std::make_tuple(importance, pruning);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_final_score", &compute_final_score_cuda, "Compute Final Score (CUDA)");
}
