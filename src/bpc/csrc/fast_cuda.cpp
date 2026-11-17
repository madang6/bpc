#include <torch/extension.h>

torch::Tensor history_accumulators_cuda(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
torch::Tensor pack_history_cuda(const torch::Tensor&, const torch::Tensor&,
                                const torch::Tensor&, int64_t);

torch::Tensor pack_history(const torch::Tensor& states, const torch::Tensor& actions,
                           const torch::Tensor& starts, int64_t history) {
  TORCH_CHECK(states.is_cuda() && actions.device() == states.device() &&
              starts.device() == states.device(), "history inputs must share CUDA device");
  TORCH_CHECK(states.dim() == 2 && actions.dim() == 2 &&
              states.size(0) == actions.size(0) && starts.dim() == 1 && history > 0,
              "invalid history dimensions");
  TORCH_CHECK(states.scalar_type() == actions.scalar_type() &&
              (states.scalar_type() == torch::kFloat64 || states.scalar_type() == torch::kFloat32) &&
              starts.scalar_type() == torch::kInt64, "invalid history dtypes");
  TORCH_CHECK(states.is_contiguous() && actions.is_contiguous() && starts.is_contiguous(),
              "history inputs must be contiguous");
  return pack_history_cuda(states, actions, starts, history);
}

torch::Tensor history_accumulators(
    const torch::Tensor& windows,
    const torch::Tensor& lengths,
    const torch::Tensor& candidates,
    const torch::Tensor& states,
    const torch::Tensor& ages,
    const torch::Tensor& weights) {
  TORCH_CHECK(windows.is_cuda() && lengths.is_cuda() && candidates.is_cuda() &&
              states.is_cuda() && ages.is_cuda() && weights.is_cuda(),
              "history inputs must be CUDA");
  TORCH_CHECK(windows.scalar_type() == torch::kFloat64 &&
              states.scalar_type() == torch::kFloat64 &&
              weights.scalar_type() == torch::kFloat64,
              "history values must be float64");
  TORCH_CHECK(lengths.scalar_type() == torch::kInt64 &&
              candidates.scalar_type() == torch::kInt64 &&
              ages.scalar_type() == torch::kInt64,
              "history indices must be int64");
  TORCH_CHECK(windows.is_contiguous() && lengths.is_contiguous() &&
              candidates.is_contiguous() && states.is_contiguous() &&
              ages.is_contiguous() && weights.is_contiguous(),
              "history inputs must be contiguous");
  TORCH_CHECK(windows.dim() == 3 && states.dim() == 2 && candidates.dim() == 2 &&
              windows.size(0) == candidates.size(0) &&
              windows.size(2) == states.size(1) &&
              lengths.sizes() == torch::IntArrayRef({windows.size(0)}) &&
              ages.sizes() == torch::IntArrayRef({states.size(0)}) &&
              weights.sizes() == torch::IntArrayRef({windows.size(1)}),
              "inconsistent history shapes");
  TORCH_CHECK(windows.size(1) <= 10, "history lookback must not exceed 10");
  TORCH_CHECK(windows.device() == lengths.device() &&
              windows.device() == candidates.device() &&
              windows.device() == states.device() && windows.device() == ages.device() &&
              windows.device() == weights.device(), "history inputs must share a device");
  return history_accumulators_cuda(
      windows, lengths, candidates, states, ages, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack_history", &pack_history, "Fused action-first Hankel gathering (CUDA)");
  module.def("history_accumulators", &history_accumulators,
             "Fused temporal-history accumulation (CUDA)");
}
