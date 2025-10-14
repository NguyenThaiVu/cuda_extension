#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

__global__ void dot_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           long long N) {
    float local = 0.0f;
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)blockDim.x * gridDim.x) {
        local += a[i] * b[i];
    }
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, smem[0]);
}

torch::Tensor dot_forward(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
              "float32 only (minimal example)");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");
  TORCH_CHECK(a.numel() == b.numel(), "Size mismatch");

  auto out = torch::zeros({}, a.options().dtype(torch::kFloat));
  const long long N = a.numel();
  const int threads = 256;
  const int blocks = (int)std::min<long long>((N + threads - 1) / threads, 4096LL);

  auto stream = at::cuda::getCurrentCUDAStream();
  dot_kernel<<<blocks, threads, 0, stream.stream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), N);
  return out;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dot_forward", &dot_forward, "Dot product (CUDA)");
}
