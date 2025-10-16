// bmm_with_streams.cu
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <algorithm>

int main() {
  // One GEMM size (same for all batches)
  int M = 128, N = 256, K = 64;
  int batch_count = 16;                 // number of independent GEMMs

  using Layout = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::Gemm<
      float, Layout,   // A
      float, Layout,   // B
      float, Layout    // C/D
  >;

  // Concatenate batches back-to-back in memory
  size_t a_stride = size_t(M) * K;      // elements between A_i and A_{i+1}
  size_t b_stride = size_t(K) * N;
  size_t c_stride = size_t(M) * N;

  std::vector<float> hA(a_stride * batch_count, 1.0f);  // A filled with 1
  std::vector<float> hB(b_stride * batch_count, 2.0f);  // B filled with 2
  std::vector<float> hC(c_stride * batch_count, 0.0f);  // C zeros

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  cudaMalloc(&dA, sizeof(float) * a_stride * batch_count);
  cudaMalloc(&dB, sizeof(float) * b_stride * batch_count);
  cudaMalloc(&dC, sizeof(float) * c_stride * batch_count);

  // Copy once (synchronous; we launch GEMMs after this)
  cudaMemcpy(dA, hA.data(), sizeof(float) * a_stride * batch_count, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), sizeof(float) * b_stride * batch_count, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), sizeof(float) * c_stride * batch_count, cudaMemcpyHostToDevice);

  // Small pool of streams (don’t oversubscribe; 4–8 is usually good)
  int num_streams = std::min(batch_count, 8);
  std::vector<cudaStream_t> streams(num_streams);
  for (int s = 0; s < num_streams; ++s) cudaStreamCreate(&streams[s]);

  // Problem + LDs (ColumnMajor → lda=M, ldb=K, ldc=M, ldd=M)
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  int lda = M, ldb = K, ldc = M, ldd = M;
  float alpha = 1.0f, beta = 0.0f;

  Gemm gemm;

  // Enqueue each GEMM on a stream (round-robin)
  for (int b = 0; b < batch_count; ++b) {
    cudaStream_t stream = streams[b % num_streams];

    float* Ab = dA + b * a_stride;
    float* Bb = dB + b * b_stride;
    float* Cb = dC + b * c_stride;

    typename Gemm::Arguments args(
        problem_size,
        {Ab, lda},        // A_b
        {Bb, ldb},        // B_b
        {Cb, ldc},        // C_b (input)
        {Cb, ldd},        // D_b (output)
        {alpha, beta}
    );

    auto status = gemm(args, stream);  // NOTE: pass the stream here
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "CUTLASS GEMM batch " << b << " failed: " << int(status) << "\n";
      return -1;
    }
  }

  // Wait for all streams to finish
  for (auto s : streams) cudaStreamSynchronize(s);

  // (Optional) gather results
  cudaMemcpy(hC.data(), dC, sizeof(float) * c_stride * batch_count, cudaMemcpyDeviceToHost);

  // Quick check: A=1, B=2 ⇒ each output = 2*K
  for (int b = 0; b < batch_count; ++b) {
    float val = hC[b * c_stride + 0];
    std::cout << "Batch " << b << " C[0] = " << val << " (expected " << 2.0f * K << ")\n";
  }

  // Cleanup
  for (auto s : streams) cudaStreamDestroy(s);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
