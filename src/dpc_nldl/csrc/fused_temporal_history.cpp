#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> temporal_history_cuda(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    int64_t, int64_t, bool);

std::vector<torch::Tensor> temporal_history(
    const torch::Tensor& states, const torch::Tensor& actions,
    const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured_rates, const torch::Tensor& lower_bounds,
    const torch::Tensor& mean, const torch::Tensor& scale,
    const torch::Tensor& scaled, int64_t history, int64_t state_offset,
    bool output_double) {
  TORCH_CHECK(states.is_cuda(), "history source must be CUDA");
  for (const auto& tensor : {states, actions, starts, rates, configured_rates,
                            lower_bounds, mean, scale, scaled}) {
    TORCH_CHECK(tensor.device() == states.device() && tensor.is_contiguous(),
                "temporal history inputs must be contiguous on one CUDA device");
  }
  TORCH_CHECK(states.dim() == 2 && actions.dim() == 2 && states.size(1) > 0 &&
              actions.size(1) > 0 && starts.dim() == 1 && history > 0 && state_offset >= 0,
              "invalid temporal history dimensions");
  TORCH_CHECK((states.scalar_type() == torch::kFloat32 || states.scalar_type() == torch::kFloat64) &&
              actions.scalar_type() == states.scalar_type() && mean.scalar_type() == states.scalar_type() &&
              scale.scalar_type() == states.scalar_type(), "invalid source/scaling precision");
  TORCH_CHECK(starts.scalar_type() == torch::kInt64 && lower_bounds.scalar_type() == torch::kInt64 &&
              configured_rates.scalar_type() == torch::kFloat64 && scaled.scalar_type() == torch::kBool &&
              (rates.scalar_type() == torch::kFloat32 || rates.scalar_type() == torch::kFloat64),
              "invalid temporal index/rate dtypes");
  TORCH_CHECK(rates.sizes() == starts.sizes() && lower_bounds.sizes() == starts.sizes() &&
              configured_rates.dim() == 1 && mean.dim() == 1 && scale.dim() == 1 && scaled.dim() == 1 &&
              mean.numel() == actions.size(1) && scale.numel() == actions.size(1) &&
              scaled.numel() == actions.size(1), "inconsistent temporal history dimensions");
  return temporal_history_cuda(states, actions, starts, rates, configured_rates,
                               lower_bounds, mean, scale, scaled, history, state_offset, output_double);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("temporal_history", &temporal_history,
             "Exact mixed-rate action-first history gather (CUDA)");
}
