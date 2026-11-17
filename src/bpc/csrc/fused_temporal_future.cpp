#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> temporal_future_cuda(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, int64_t, int64_t, int64_t, int64_t, bool);

std::vector<torch::Tensor> temporal_future(
    const torch::Tensor& source, const torch::Tensor& native,
    const torch::Tensor& columns, const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured, const torch::Tensor& stops,
    const torch::Tensor& mean, const torch::Tensor& scale, const torch::Tensor& scaled,
    int64_t history, int64_t horizon, int64_t shift, int64_t channels, bool states) {
  TORCH_CHECK(source.is_cuda(), "future source must be CUDA");
  for (const auto& tensor : {source, columns, starts, rates, configured, stops, mean, scale, scaled})
    TORCH_CHECK(tensor.device() == source.device() && tensor.is_contiguous(),
                "future inputs must be contiguous on one CUDA device");
  TORCH_CHECK(native.device() == source.device() && native.dim() == 2 && source.dim() == 2 &&
              history > 0 && horizon > 0 && shift >= 0 && channels > 0 && channels <= source.size(1),
              "invalid future dimensions");
  TORCH_CHECK((source.scalar_type() == torch::kFloat32 || source.scalar_type() == torch::kFloat64) &&
              (native.scalar_type() == torch::kFloat32 || native.scalar_type() == torch::kFloat64) &&
              (rates.scalar_type() == torch::kFloat32 || rates.scalar_type() == torch::kFloat64),
              "unsupported future precision");
  TORCH_CHECK(columns.scalar_type() == torch::kInt64 && starts.scalar_type() == torch::kInt64 &&
              stops.scalar_type() == torch::kInt64 && configured.scalar_type() == torch::kFloat64 &&
              mean.scalar_type() == source.scalar_type() && scale.scalar_type() == source.scalar_type() &&
              scaled.scalar_type() == torch::kBool, "invalid future metadata dtypes");
  TORCH_CHECK(columns.dim() == 1 && starts.sizes() == columns.sizes() && rates.sizes() == columns.sizes() &&
              stops.sizes() == columns.sizes() && configured.dim() == 1, "invalid future metadata shapes");
  TORCH_CHECK(states || (native.size(0) == horizon*channels && mean.numel() == channels &&
              scale.numel() == channels && scaled.numel() == channels), "invalid native action/scaling shapes");
  TORCH_CHECK(!states || native.scalar_type() == source.scalar_type(), "invalid state output precision");
  return temporal_future_cuda(source, native, columns, starts, rates, configured,
      stops, mean, scale, scaled, history, horizon, shift, channels, states);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("temporal_future", &temporal_future, "Exact mixed-rate action/state futures (CUDA)");
}
