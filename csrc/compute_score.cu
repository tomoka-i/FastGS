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
    bool densify) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // スコアの計算（元のコードの正規化ロジックを並列化）
    float counts = metric_counts[idx];
    float score = (counts > 0) ? (metric_score[idx] / counts) : 0.0f;

    importance[idx] = score;
    pruning[idx] = densify ? score * 0.5f : score * 1.5f;
}

// C++ラッパー
std::tuple<torch::Tensor, torch::Tensor> compute_final_score_cuda(
    torch::Tensor metric_score,
    torch::Tensor metric_counts,
    bool densify) 
{
    int N = metric_score.size(0);
    auto importance = torch::zeros({N, 1}, metric_score.options());
    auto pruning = torch::zeros({N, 1}, metric_score.options());

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    compute_final_score_kernel<<<grid, block>>>(
        metric_score.data_ptr<float>(),
        metric_counts.data_ptr<float>(),
        importance.data_ptr<float>(),
        pruning.data_ptr<float>(),
        N,
        densify
    );

    return std::make_tuple(importance, pruning);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_final_score", &compute_final_score_cuda, "Compute Final Score (CUDA)");
}