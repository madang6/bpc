#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarps = kThreads / kWarp;

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
#pragma unroll
  for (int offset = kWarp / 2; offset; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void history_kernel(
    const double* __restrict__ windows,
    const int64_t* __restrict__ lengths,
    const int64_t* __restrict__ candidates,
    const double* __restrict__ states,
    const int64_t* __restrict__ ages,
    const double* __restrict__ weights,
    int lookback, int candidates_count, int width,
    double* __restrict__ output) {
  extern __shared__ double shared[];
  double* query = shared;
  double* accumulated = query + width;
  const int batch = blockIdx.x;
  const int thread = threadIdx.x;
  const int lane = thread % kWarp;
  const int warp = thread / kWarp;
  const int64_t* selected = candidates + batch * candidates_count;
  for (int candidate = thread; candidate < candidates_count; candidate += kThreads)
    accumulated[candidate] = 0;
  __syncthreads();
  for (int offset = 0; offset < lookback; ++offset) {
    for (int feature = thread; feature < width; feature += kThreads)
      query[feature] = windows[(batch * lookback + lookback - 1 - offset) * width + feature];
    __syncthreads();
    for (int candidate = warp; candidate < candidates_count; candidate += kWarps) {
      const int64_t row = selected[candidate];
      if (offset < lengths[batch] && offset <= ages[row]) {
        double square = 0;
        for (int feature = lane; feature < width; feature += kWarp) {
          const double difference = states[(row - offset) * width + feature] - query[feature];
          square = __dadd_rn(square, __dmul_rn(difference, difference));
        }
        square = warp_sum(square);
        if (lane == 0)
          accumulated[candidate] = __dadd_rn(
              accumulated[candidate], __dmul_rn(weights[offset], __dsqrt_rn(square)));
      }
    }
    __syncthreads();
  }
  for (int candidate = thread; candidate < candidates_count; candidate += kThreads)
    output[batch * candidates_count + candidate] = accumulated[candidate];
}

template <typename T>
__global__ void pack_history_kernel(const T* states, const T* actions,
                                    const int64_t* starts, int history,
                                    int state_dim, int action_dim, T* output) {
  const int width = history * (state_dim + action_dim);
  const int64_t start = starts[blockIdx.x];
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const bool action = col < history * action_dim;
    const int offset = action ? col : col - history * action_dim;
    const int dim = action ? action_dim : state_dim;
    output[int64_t(blockIdx.x) * width + col] = (action ? actions : states)[
        (start + offset / dim) * dim + offset % dim];
  }
}

}  // namespace

torch::Tensor pack_history_cuda(const torch::Tensor& states, const torch::Tensor& actions,
                                const torch::Tensor& starts, int64_t history) {
  c10::cuda::CUDAGuard guard(states.device());
  auto output = torch::empty({starts.size(0), history * (states.size(1) + actions.size(1))},
                             states.options());
  if (starts.numel()) AT_DISPATCH_FLOATING_TYPES(states.scalar_type(), "pack_history", [&] {
    pack_history_kernel<<<starts.size(0), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        states.data_ptr<scalar_t>(), actions.data_ptr<scalar_t>(), starts.data_ptr<int64_t>(),
        history, states.size(1), actions.size(1), output.data_ptr<scalar_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor history_accumulators_cuda(
    const torch::Tensor& windows,
    const torch::Tensor& lengths,
    const torch::Tensor& candidates,
    const torch::Tensor& states,
    const torch::Tensor& ages,
    const torch::Tensor& weights) {
  c10::cuda::CUDAGuard guard(states.device());
  auto output = torch::empty(candidates.sizes(), states.options());
  const size_t shared = (states.size(1) + candidates.size(1)) * sizeof(double);
  history_kernel<<<windows.size(0), kThreads, shared,
                   at::cuda::getCurrentCUDAStream()>>>(
      windows.data_ptr<double>(), lengths.data_ptr<int64_t>(),
      candidates.data_ptr<int64_t>(), states.data_ptr<double>(),
      ages.data_ptr<int64_t>(), weights.data_ptr<double>(), windows.size(1),
      candidates.size(1), states.size(1), output.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
