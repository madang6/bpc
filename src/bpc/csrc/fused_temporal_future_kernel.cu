#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <vector>

namespace {

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

__device__ __forceinline__ double position(int64_t endpoint, double rate, int step, int shift) {
  return __dadd_rn(__dadd_rn(static_cast<double>(endpoint), __dmul_rn(rate, step)), shift);
}

template <typename T, typename Rate, typename Output>
__global__ void temporal_future_kernel(
    const T* source, const Output* native, const int64_t* columns, const int64_t* starts,
    const Rate* rates, const double* configured, int rate_count, const int64_t* stops,
    const T* mean, const T* scale, const bool* scaled, int64_t source_rows, int source_width,
    int64_t native_columns, int64_t native_stride0, int64_t native_stride1,
    int history, int horizon, int shift, int channels, bool states,
    Output* output, int* error) {
  __shared__ double rate;
  __shared__ int invalid;
  const int64_t row = blockIdx.x;
  const int64_t endpoint = starts[row]+history-1;
  if (threadIdx.x == 0) {
    rate = 1.;
    invalid = 0;
    // The original function first gathers every native column, even where a
    // slower valid time map later overwrites it. Preserve that access/error.
    if (states) {
      if (endpoint+1+shift < 0 || endpoint+horizon+shift >= source_rows) invalid |= 4;
    } else if (columns[row] < 0 || columns[row] >= native_columns) invalid |= 4;
    for (int i = 0; i < rate_count; ++i) {
      if (rates[row] == static_cast<Rate>(configured[i])) {
        rate = configured[i];
        const double first = position(endpoint, rate, 1, shift);
        const double last = position(endpoint, rate, horizon, shift);
        if (!isfinite(first) || !isfinite(last)) invalid |= 2;
        else {
          // Check all matching configured rates, including intermediate
          // overwrites when distinct doubles round to the same FP32 rate.
          if (ceil(last) >= static_cast<double>(stops[row])) invalid |= 1;
          if (first < 0. || ceil(last) >= source_rows) invalid |= 2;
        }
      }
    }
    if (invalid) atomicOr(error, invalid);
  }
  __syncthreads();
  if (invalid) return;
  const int width = horizon*channels;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    const int step = column/channels+1, component = column%channels;
    if (rate == 1. && !states) {
      // Never reconstruct native uf from action rows: preserve its exact
      // stored values, output dtype, and arbitrary positive tensor strides.
      output[row*width+column] = native[column*native_stride0+columns[row]*native_stride1];
      continue;
    }
    T value;
    if (rate == 1.) {
      value = source[(endpoint+step+shift)*source_width+component];
    } else {
      const double pos = position(endpoint, rate, step, shift);
      const int64_t lower = static_cast<int64_t>(floor(pos));
      const int64_t upper = static_cast<int64_t>(ceil(pos));
      const T weight = static_cast<T>(__dsub_rn(pos, static_cast<double>(lower)));
      const T left = source[lower*source_width+component];
      const T right = source[upper*source_width+component];
      value = Arithmetic<T>::add(left, Arithmetic<T>::mul(weight, Arithmetic<T>::sub(right, left)));
      if (!states && scaled[component]) {
        value = Arithmetic<T>::mul(value, scale[component]);
        value = Arithmetic<T>::add(value, mean[component]);
        value = Arithmetic<T>::mul(value, static_cast<T>(rate));
        value = Arithmetic<T>::sub(value, mean[component]);
        value = Arithmetic<T>::div(value, scale[component]);
      }
    }
    output[row*width+column] = static_cast<Output>(value);
  }
}

template <typename T, typename Rate, typename Output>
void launch(const torch::Tensor& source, const torch::Tensor& native,
    const torch::Tensor& columns, const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured, const torch::Tensor& stops,
    const torch::Tensor& mean, const torch::Tensor& scale, const torch::Tensor& scaled,
    int64_t history, int64_t horizon, int64_t shift, int64_t channels, bool states,
    torch::Tensor& output, torch::Tensor& error) {
  temporal_future_kernel<T, Rate, Output><<<columns.numel(), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
    source.data_ptr<T>(), native.data_ptr<Output>(), columns.data_ptr<int64_t>(), starts.data_ptr<int64_t>(),
    rates.data_ptr<Rate>(), configured.data_ptr<double>(), configured.numel(), stops.data_ptr<int64_t>(),
    mean.data_ptr<T>(), scale.data_ptr<T>(), scaled.data_ptr<bool>(), source.size(0), source.size(1),
    native.size(1), native.stride(0), native.stride(1), history, horizon, shift, channels, states,
    output.data_ptr<Output>(), error.data_ptr<int>());
}

template <typename T, typename Rate>
void output_dispatch(const torch::Tensor& source, const torch::Tensor& native,
    const torch::Tensor& columns, const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured, const torch::Tensor& stops,
    const torch::Tensor& mean, const torch::Tensor& scale, const torch::Tensor& scaled,
    int64_t history, int64_t horizon, int64_t shift, int64_t channels, bool states,
    torch::Tensor& output, torch::Tensor& error) {
  if (native.scalar_type() == torch::kFloat64)
    launch<T, Rate, double>(source, native, columns, starts, rates, configured, stops,
                           mean, scale, scaled, history, horizon, shift, channels, states, output, error);
  else
    launch<T, Rate, float>(source, native, columns, starts, rates, configured, stops,
                          mean, scale, scaled, history, horizon, shift, channels, states, output, error);
}

}  // namespace

std::vector<torch::Tensor> temporal_future_cuda(
    const torch::Tensor& source, const torch::Tensor& native,
    const torch::Tensor& columns, const torch::Tensor& starts, const torch::Tensor& rates,
    const torch::Tensor& configured, const torch::Tensor& stops,
    const torch::Tensor& mean, const torch::Tensor& scale, const torch::Tensor& scaled,
    int64_t history, int64_t horizon, int64_t shift, int64_t channels, bool states) {
  c10::cuda::CUDAGuard guard(source.device());
  auto output = torch::empty({columns.numel(), horizon*channels}, source.options().dtype(native.scalar_type()));
  auto error = torch::zeros({1}, source.options().dtype(torch::kInt32));
  if (columns.numel()) {
    AT_DISPATCH_FLOATING_TYPES(source.scalar_type(), "temporal_future", [&] {
      if (rates.scalar_type() == torch::kFloat64)
        output_dispatch<scalar_t, double>(source, native, columns, starts, rates, configured,
            stops, mean, scale, scaled, history, horizon, shift, channels, states, output, error);
      else
        output_dispatch<scalar_t, float>(source, native, columns, starts, rates, configured,
            stops, mean, scale, scaled, history, horizon, shift, channels, states, output, error);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, error};
}
