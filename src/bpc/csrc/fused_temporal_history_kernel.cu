#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <vector>

namespace {

// PyTorch executes these as separate elementwise operations. Explicit rounded
// intrinsics retain those rounding points even when the shared loader enables
// contraction for other kernels. No fast-math or reciprocal approximation.
template <typename T> struct Arithmetic;
template <> struct Arithmetic<float> {
  __device__ static float add(float a, float b) { return __fadd_rn(a, b); }
  __device__ static float sub(float a, float b) { return __fsub_rn(a, b); }
  __device__ static float mul(float a, float b) { return __fmul_rn(a, b); }
  __device__ static float div(float a, float b) { return __fdiv_rn(a, b); }
};
template <> struct Arithmetic<double> {
  __device__ static double add(double a, double b) { return __dadd_rn(a, b); }
  __device__ static double sub(double a, double b) { return __dsub_rn(a, b); }
  __device__ static double mul(double a, double b) { return __dmul_rn(a, b); }
  __device__ static double div(double a, double b) { return __ddiv_rn(a, b); }
};

template <typename T, typename Rate, typename Output>
__global__ void temporal_history_kernel(
    const T* states, const T* actions, const int64_t* starts, const Rate* rates,
    const double* configured, int rate_count, const int64_t* lower_bounds,
    const T* means, const T* scales, const bool* scaled,
    int64_t state_rows, int64_t action_rows, int state_dim, int action_dim,
    int history, int state_offset, Output* output, int* error) {
  __shared__ double rate;
  __shared__ int invalid;
  const int64_t row = blockIdx.x;
  const int64_t start = starts[row];
  const int64_t endpoint = start + history - 1;
  if (threadIdx.x == 0) {
    // Compare in the rate tensor's dtype, then interpolate with the configured
    // Python double. Later matching rates overwrite earlier ones, as before.
    invalid = 0;
    if (start < 0 || endpoint >= action_rows || start+state_offset < 0 ||
        endpoint+state_offset >= state_rows) invalid |= 4;
    rate = 1.;
    for (int i = 0; i < rate_count; ++i) {
      if (rates[row] == static_cast<Rate>(configured[i])) {
        rate = configured[i];
        const double first = __dadd_rn(static_cast<double>(endpoint), __dmul_rn(rate, 1. - history));
        // A rounded rate tensor can match several Python rates. The original
        // loop checks each intermediate gather, even if a later one overwrites
        // it, so preserve every matched rate's boundary exception.
        if (!isfinite(first) || !isfinite(rate)) invalid |= 2;
        else {
          if (floor(first) < static_cast<double>(lower_bounds[row])) invalid |= 1;
          if (first < 0.) invalid |= 2;
        }
      }
    }
    const double first = rate == 1. ? static_cast<double>(start) :
        __dadd_rn(static_cast<double>(endpoint), __dmul_rn(rate, 1. - history));
    if (!isfinite(first) || !isfinite(rate)) invalid |= 2;
    else {
      if (rate != 1. && floor(first) < static_cast<double>(lower_bounds[row])) invalid |= 1;
      if (first < 0. || endpoint < 0 || endpoint >= action_rows ||
          endpoint + state_offset >= state_rows) invalid |= 2;
    }
    if (invalid) atomicOr(error, invalid);
  }
  __syncthreads();
  if (invalid) return;  // Bounds failures never perform speculative source reads.
  const int64_t width = int64_t(history) * (state_dim + action_dim);
  for (int64_t column = threadIdx.x; column < width; column += blockDim.x) {
    const bool is_action = column < int64_t(history)*action_dim;
    const int dim = is_action ? action_dim : state_dim;
    const int64_t field = is_action ? column : column - int64_t(history)*action_dim;
    const int time = field / dim, component = field % dim;
    const T* source = is_action ? actions : states;
    const int shift = is_action ? 0 : state_offset;
    T value;
    if (rate == 1.) {
      value = source[(start + time + shift)*dim + component];
    } else {
      const double offset = __dadd_rn(__dsub_rn(static_cast<double>(time), history), 1.);
      const double position = __dadd_rn(static_cast<double>(endpoint), __dmul_rn(rate, offset));
      const int64_t lower = static_cast<int64_t>(floor(position));
      const int64_t upper = static_cast<int64_t>(ceil(position));
      const T weight = static_cast<T>(__dsub_rn(position, static_cast<double>(lower)));
      const T left = source[(lower + shift)*dim + component];
      const T right = source[(upper + shift)*dim + component];
      value = Arithmetic<T>::add(left, Arithmetic<T>::mul(weight, Arithmetic<T>::sub(right, left)));
      if (is_action && scaled[component]) {
        value = Arithmetic<T>::mul(value, scales[component]);
        value = Arithmetic<T>::add(value, means[component]);
        value = Arithmetic<T>::mul(value, static_cast<T>(rate));
        value = Arithmetic<T>::sub(value, means[component]);
        value = Arithmetic<T>::div(value, scales[component]);
      }
    }
    output[row*width + column] = static_cast<Output>(value);
  }
}

template <typename T, typename Rate, typename Output>
void launch(const torch::Tensor& states, const torch::Tensor& actions,
    const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured, const torch::Tensor& lower_bounds,
    const torch::Tensor& mean, const torch::Tensor& scale, const torch::Tensor& scaled,
    int64_t history, int64_t state_offset, torch::Tensor& output, torch::Tensor& error) {
  temporal_history_kernel<T, Rate, Output><<<starts.numel(), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
    states.data_ptr<T>(), actions.data_ptr<T>(), starts.data_ptr<int64_t>(), rates.data_ptr<Rate>(),
    configured.data_ptr<double>(), configured.numel(), lower_bounds.data_ptr<int64_t>(),
    mean.data_ptr<T>(), scale.data_ptr<T>(), scaled.data_ptr<bool>(),
    states.size(0), actions.size(0), states.size(1), actions.size(1), history, state_offset,
    output.data_ptr<Output>(), error.data_ptr<int>());
}

template <typename T, typename Rate>
void output_dispatch(const torch::Tensor& states, const torch::Tensor& actions,
    const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured, const torch::Tensor& lower_bounds,
    const torch::Tensor& mean, const torch::Tensor& scale, const torch::Tensor& scaled,
    int64_t history, int64_t state_offset, torch::Tensor& output, torch::Tensor& error) {
  if (output.scalar_type() == torch::kFloat64)
    launch<T, Rate, double>(states, actions, starts, rates, configured, lower_bounds,
                           mean, scale, scaled, history, state_offset, output, error);
  else
    launch<T, Rate, float>(states, actions, starts, rates, configured, lower_bounds,
                          mean, scale, scaled, history, state_offset, output, error);
}

}  // namespace

std::vector<torch::Tensor> temporal_history_cuda(
    const torch::Tensor& states, const torch::Tensor& actions,
    const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured, const torch::Tensor& lower_bounds,
    const torch::Tensor& mean, const torch::Tensor& scale, const torch::Tensor& scaled,
    int64_t history, int64_t state_offset, bool output_double) {
  c10::cuda::CUDAGuard guard(states.device());
  auto output = torch::empty({starts.numel(), history*(states.size(1)+actions.size(1))},
      states.options().dtype(output_double ? torch::kFloat64 : torch::kFloat32));
  auto error = torch::zeros({1}, states.options().dtype(torch::kInt32));
  if (starts.numel()) {
    AT_DISPATCH_FLOATING_TYPES(states.scalar_type(), "temporal_history", [&] {
      if (rates.scalar_type() == torch::kFloat64)
        output_dispatch<scalar_t, double>(states, actions, starts, rates, configured, lower_bounds,
                                         mean, scale, scaled, history, state_offset, output, error);
      else
        output_dispatch<scalar_t, float>(states, actions, starts, rates, configured, lower_bounds,
                                        mean, scale, scaled, history, state_offset, output, error);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, error};
}
