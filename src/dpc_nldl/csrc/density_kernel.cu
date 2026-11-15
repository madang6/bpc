#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {
// Preserve the two reference expressions' addition order and rounding boundaries.
__device__ double squared_distance(double product, double query, double bank) {
  return __dadd_rn(__dadd_rn(__dmul_rn(product, -2.), query), bank);
}
__device__ float squared_distance(float product, float query, float bank) {
  return __fadd_rn(__fadd_rn(__fmul_rn(product, -2.f), bank), query);
}

template <typename scalar_t>
__global__ void postprocess(scalar_t* output, const scalar_t* product,
    const scalar_t* query_norm, const scalar_t* bank_norm,
    const int64_t* starts, const int64_t* rows, const int64_t* demos,
    const int64_t* trajectories, const int64_t* first,
    const bool* valid, const bool* allowed, int32_t* invalid,
    int64_t elements, int64_t width, int64_t history, int64_t last_offset,
    bool initial, bool final) {
  for (int64_t position = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       position < elements; position += int64_t(gridDim.x)*blockDim.x) {
    const int64_t row = position/width, column = position%width;
    scalar_t value = squared_distance(product[position], query_norm[row], bank_norm[column]);
    if (invalid && !isfinite(value)) atomicOr(invalid+row, 1);
    value = isnan(value) ? value : max(value, scalar_t(0));
    // PyTorch timestamp arithmetic wraps even for extreme int64 values.
    const int64_t last = static_cast<int64_t>(
        static_cast<uint64_t>(starts[column])+static_cast<uint64_t>(last_offset));
    if (valid && (!valid[column] || (rows && first[column] <= rows[row]
                                         && rows[row] <= last)))
      value = INFINITY;
    if (!initial) {
      const scalar_t previous = output[position];
      value = isnan(previous) ? previous : (isnan(value) ? value : min(previous, value));
    }
    const int64_t lower = rows ? static_cast<int64_t>(
        static_cast<uint64_t>(rows[row])-static_cast<uint64_t>(history)) : 0;
    if (final && ((rows && lower <= starts[column] && starts[column] <= rows[row])
                  || (demos && trajectories[column] == demos[row])
                  || (allowed && !allowed[position])))
      value = INFINITY;
    output[position] = value;
  }
}
}

void density_cuda(torch::Tensor output, const torch::Tensor& product,
    const torch::Tensor& query_norm, const torch::Tensor& bank_norm,
    const torch::Tensor& starts, const torch::Tensor& rows, const torch::Tensor& demos,
    const torch::Tensor& trajectories, const torch::Tensor& first,
    const torch::Tensor& valid, const torch::Tensor& allowed, torch::Tensor invalid,
    int64_t history, int64_t last_offset, bool initial, bool final) {
  c10::cuda::CUDAGuard guard(output.device());
  if (!output.numel()) return;
  int blocks = std::min<int64_t>((output.numel()+255)/256, 65535);
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "density", [&] {
  postprocess<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<scalar_t>(), product.data_ptr<scalar_t>(), query_norm.data_ptr<scalar_t>(),
      bank_norm.data_ptr<scalar_t>(), starts.data_ptr<int64_t>(),
      rows.numel() ? rows.data_ptr<int64_t>() : nullptr,
      demos.numel() ? demos.data_ptr<int64_t>() : nullptr, trajectories.data_ptr<int64_t>(),
      first.numel() ? first.data_ptr<int64_t>() : nullptr,
      valid.numel() ? valid.data_ptr<bool>() : nullptr,
      allowed.numel() ? allowed.data_ptr<bool>() : nullptr,
      invalid.defined() ? invalid.data_ptr<int32_t>() : nullptr,
      output.numel(), output.size(1), history, last_offset, initial, final);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
