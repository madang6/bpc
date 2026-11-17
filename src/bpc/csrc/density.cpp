#include <torch/extension.h>

void density_cuda(torch::Tensor, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    torch::Tensor, int64_t, int64_t, bool, bool);

void density(torch::Tensor output, const torch::Tensor& product,
    const torch::Tensor& query_norm, const torch::Tensor& bank_norm,
    const torch::Tensor& starts, const torch::Tensor& rows, const torch::Tensor& demos,
    const torch::Tensor& trajectories, const torch::Tensor& first,
    const torch::Tensor& valid, const torch::Tensor& allowed, torch::Tensor invalid,
    int64_t history, int64_t last_offset, bool initial, bool final) {
  TORCH_CHECK(output.is_cuda() && output.dim() == 2 && product.sizes() == output.sizes(),
              "screen products must be equally shaped CUDA matrices");
  for (const auto& tensor : {output, product, query_norm, bank_norm, starts, rows, demos,
                            trajectories, first, valid, allowed})
    TORCH_CHECK(tensor.device() == output.device() && tensor.is_contiguous(),
                "screen tensors must be contiguous on one CUDA device");
  const auto precision = invalid.defined() ? torch::kFloat32 : torch::kFloat64;
  for (const auto& tensor : {output, product, query_norm, bank_norm})
    TORCH_CHECK(tensor.scalar_type() == precision,
                invalid.defined() ? "screen arithmetic must be FP32" : "density arithmetic must be FP64");
  for (const auto& tensor : {starts, rows, demos, trajectories, first})
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, "screen indices must be int64");
  TORCH_CHECK(valid.scalar_type() == torch::kBool && allowed.scalar_type() == torch::kBool,
              "screen masks must be boolean");
  if (invalid.defined())
    TORCH_CHECK(invalid.device() == output.device() && invalid.is_contiguous() &&
                invalid.scalar_type() == torch::kInt32, "screen error flags must be contiguous CUDA int32");
  auto batch = output.size(0), width = output.size(1);
  TORCH_CHECK(query_norm.numel() == batch && bank_norm.numel() == width &&
              starts.numel() == width && trajectories.numel() == width &&
              (!invalid.defined() || invalid.numel() == batch) &&
              (!rows.numel() || rows.numel() == batch) &&
              (!demos.numel() || demos.numel() == batch) &&
              (!first.numel() || first.numel() == width) &&
              valid.numel() == first.numel() &&
              (!allowed.numel() || allowed.sizes() == output.sizes()),
              "inconsistent screen dimensions");
  density_cuda(output, product, query_norm, bank_norm, starts, rows, demos,
                     trajectories, first, valid, allowed, invalid,
                     history, last_offset, initial, final);
}

// Keep both call signatures: only the FP32 screen accepts per-row error flags.
void exact_density(torch::Tensor output, const torch::Tensor& product,
    const torch::Tensor& query_norm, const torch::Tensor& bank_norm,
    const torch::Tensor& starts, const torch::Tensor& rows, const torch::Tensor& demos,
    const torch::Tensor& trajectories, const torch::Tensor& first,
    const torch::Tensor& valid, const torch::Tensor& allowed,
    int64_t history, int64_t last_offset, bool initial, bool final) {
  density(output, product, query_norm, bank_norm, starts, rows, demos,
          trajectories, first, valid, allowed, {}, history, last_offset, initial, final);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("postprocess", &exact_density, "Exact FP64 density distances (CUDA)");
  module.def("postprocess", &density, "FP32 density screen and finite flags (CUDA)");
}
