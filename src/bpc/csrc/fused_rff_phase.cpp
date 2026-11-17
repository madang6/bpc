#include <torch/extension.h>

void phase_cosine_cuda(torch::Tensor phase, const torch::Tensor& bias,
                       const torch::Tensor& projected);

void phase_cosine(torch::Tensor phase, const torch::Tensor& bias,
                  const torch::Tensor& projected) {
  TORCH_CHECK(phase.is_cuda() && phase.dim() == 3 && phase.is_contiguous(),
              "phase must be contiguous CUDA [batch, support, features]");
  TORCH_CHECK((phase.scalar_type() == torch::kFloat32 || phase.scalar_type() == torch::kFloat64) &&
              bias.scalar_type() == phase.scalar_type() && projected.scalar_type() == phase.scalar_type(),
              "RFF pointwise inputs must have the same float32/float64 dtype");
  TORCH_CHECK(bias.device() == phase.device() && projected.device() == phase.device() &&
              bias.dim() == 1 && bias.size(0) == phase.size(2) &&
              projected.dim() == 3 && projected.size(0) == phase.size(0) &&
              projected.size(1) == 1 && projected.size(2) == phase.size(2),
              "invalid RFF pointwise shapes/devices");
  phase_cosine_cuda(phase, bias, projected);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("phase_cosine_", &phase_cosine, "Rounded add/subtract and normal CUDA cosine in place");
}
