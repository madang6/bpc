#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <algorithm>
#include <cmath>

namespace {
template <typename T> struct Pointwise;
template <> struct Pointwise<float> {
  __device__ static float apply(float phase, float bias, float query) {
    return ::cosf(__fsub_rn(__fadd_rn(phase, bias), query));
  }
};
template <> struct Pointwise<double> {
  __device__ static double apply(double phase, double bias, double query) {
    return ::cos(__dsub_rn(__dadd_rn(phase, bias), query));
  }
};

template <typename T>
__global__ void phase_cosine_kernel(T* phase, const T* bias, const T* projected,
    int64_t elements, int64_t support, int64_t features, int64_t bias_stride,
    int64_t query_batch_stride, int64_t query_feature_stride) {
  for (int64_t index = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       index < elements; index += int64_t(gridDim.x)*blockDim.x) {
    const int64_t feature = index%features, batch = index/(support*features);
    phase[index] = Pointwise<T>::apply(phase[index], bias[feature*bias_stride],
        projected[batch*query_batch_stride+feature*query_feature_stride]);
  }
}
}

void phase_cosine_cuda(torch::Tensor phase, const torch::Tensor& bias,
                       const torch::Tensor& projected) {
  c10::cuda::CUDAGuard guard(phase.device());
  const int64_t count = phase.numel();
  if (!count) return;
  const int blocks = static_cast<int>(std::min<int64_t>((count+255)/256, 4096));
  AT_DISPATCH_FLOATING_TYPES(phase.scalar_type(), "phase_cosine", [&] {
    phase_cosine_kernel<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        phase.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), projected.data_ptr<scalar_t>(),
        count, phase.size(1), phase.size(2), bias.stride(0), projected.stride(0), projected.stride(2));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
