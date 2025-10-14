#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
  // (M x K) * (K x N) = (M x N)
  int M = 128, N = 256, K = 64;

  // Host buffers (column-major because that’s CUTLASS default in this setup)
  using Layout = cutlass::layout::ColumnMajor;
  std::vector<float> hA(M*K, 1.0f);     // A filled with 1
  std::vector<float> hB(K*N, 2.0f);     // B filled with 2
  std::vector<float> hC(M*N, 0.0f);     // C zeros

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  cudaMalloc(&dA, sizeof(float)*M*K);
  cudaMalloc(&dB, sizeof(float)*K*N);
  cudaMalloc(&dC, sizeof(float)*M*N);

  cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);

  using Gemm = cutlass::gemm::device::Gemm<
      float, Layout,   // A
      float, Layout,   // B
      float, Layout    // C/D
  >;

  Gemm gemm;
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  float alpha = 1.0f, beta = 0.0f;

  typename Gemm::Arguments args(
      problem_size,
      {dA, M},    // lda = leading dimension of A (since column-major, lda = rows = M)
      {dB, K},    // ldb = K
      {dC, M},    // ldc = M
      {dC, M},    // ldd = M
      {alpha, beta}
  );

  auto status = gemm(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS SGEMM failed: " << int(status) << "\n";
    return -1;
  }

  cudaMemcpy(hC.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  // Since A=1, B=2 ⇒ each output = sum_k (1*2) = 2*K
  std::cout << "C[0] = " << hC[0] << " (expected " << 2.0f*K << ")\n";

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
