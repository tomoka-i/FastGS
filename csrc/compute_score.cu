#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void compute_l1_map_kernel(
    const float* reconstructed,
    const float* original,
    float* l1_map,
    int height,
    int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = height * width;
    if (idx >= pixels) return;

    float diff_sum = 0.0f;
    for (int c = 0; c < 3; ++c) {
        int offset = c * pixels + idx;
        diff_sum += fabsf(reconstructed[offset] - original[offset]);
    }
    l1_map[idx] = diff_sum / 3.0f;
}

__global__ void threshold_metric_map_kernel(
    const float* l1_map,
    int* metric_map,
    int pixels,
    float min_loss,
    float loss_range,
    float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;

    float normalized = loss_range > 0.0f ? (l1_map[idx] - min_loss) / loss_range : 0.0f;
    metric_map[idx] = normalized > threshold ? 1 : 0;
}

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
    pruning[idx] = score_range > 0.0f ? (metric_score[idx] - min_score) / score_range : 0.0f;
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

torch::Tensor compute_metric_map_cuda(
    torch::Tensor reconstructed,
    torch::Tensor original,
    float threshold)
{
    TORCH_CHECK(reconstructed.is_cuda(), "reconstructed must be a CUDA tensor");
    TORCH_CHECK(original.is_cuda(), "original must be a CUDA tensor");
    TORCH_CHECK(reconstructed.scalar_type() == torch::kFloat32, "reconstructed must be float32");
    TORCH_CHECK(original.scalar_type() == torch::kFloat32, "original must be float32");
    TORCH_CHECK(reconstructed.sizes() == original.sizes(), "input image sizes must match");
    TORCH_CHECK(reconstructed.dim() == 3, "expected image tensors with shape [C, H, W]");
    TORCH_CHECK(reconstructed.size(0) == 3, "expected 3-channel image tensors");

    reconstructed = reconstructed.contiguous();
    original = original.contiguous();

    int height = reconstructed.size(1);
    int width = reconstructed.size(2);
    int pixels = height * width;

    auto l1_map = torch::empty({height, width}, reconstructed.options());
    auto metric_map = torch::empty({height, width}, reconstructed.options().dtype(torch::kInt32));

    dim3 block(256);
    dim3 grid((pixels + block.x - 1) / block.x);

    compute_l1_map_kernel<<<grid, block>>>(
        reconstructed.data_ptr<float>(),
        original.data_ptr<float>(),
        l1_map.data_ptr<float>(),
        height,
        width
    );

    float min_loss = l1_map.min().item<float>();
    float max_loss = l1_map.max().item<float>();
    float loss_range = max_loss - min_loss;

    threshold_metric_map_kernel<<<grid, block>>>(
        l1_map.data_ptr<float>(),
        metric_map.data_ptr<int>(),
        pixels,
        min_loss,
        loss_range,
        threshold
    );

    return metric_map;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_final_score", &compute_final_score_cuda, "Compute Final Score (CUDA)");
    m.def("compute_metric_map", &compute_metric_map_cuda, "Compute metric map (CUDA)");
}
