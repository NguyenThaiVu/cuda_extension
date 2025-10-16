#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/util/device_memory.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>

// CUTLASS 4.x consolidated layouts here
#if __has_include(<cutlass/layout/matrix.h>)
  #include <cutlass/layout/matrix.h>
#else
  #include <cutlass/layout/row_major.h>
  #include <cutlass/layout/column_major_interleaved.h>
  #include <cutlass/layout/column_major.h>
#endif

// ----------------------------------
// Simple GPU timer 
struct GpuTimer {
  cudaEvent_t start{}, stop{};
  cudaStream_t stream{0};
  explicit GpuTimer(cudaStream_t s = 0) : stream(s) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void Start() { cudaEventRecord(start, stream); }
  void Stop()  { cudaEventRecord(stop, stream);  }
  float Elapsed() {
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};

// ----------------------------------
// INT8 x INT8 -> INT32 (column-major)
// C = alpha * A * B + beta * C
// A:[M,K], B:[K,N], C:[M,N]; all column-major
// lda=M, ldb=K, ldc=M
// ----------------------------------
cutlass::Status matmul_int8(
    int M, int N, int K,
    const int8_t* dA, int lda,
    const int8_t* dB, int ldb,
    int32_t* dC, int ldc,
    int32_t alpha = 1, int32_t beta = 0,
    cudaStream_t stream = 0) {

  // —— Optional safety: DP4A kernels like K%4==0. CUTLASS can handle tails,
  // but this check helps catch perf surprises early.
  if (K % 4 != 0) {
    std::cerr << "[warn] K=" << K << " is not a multiple of 4; INT8 DP4A may be slower.\n";
  }

  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;

  using ElementA   = int8_t;
  using ElementB   = int8_t;
  using ElementC   = int32_t;   // output / D
  using ElementAcc = int32_t;   // accumulator

  // SIMT epilogue must be scalar (ElementsPerAccess = 1)
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAcc, ElementAcc>;

  // SIMT/DP4A kernel (robust & easy to build)
  using Gemm = cutlass::gemm::device::Gemm<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAcc,
      cutlass::arch::OpClassSimt,            // DP4A path
      cutlass::arch::Sm86,                   // Ada (change if you target other arch)
      cutlass::gemm::GemmShape<128,128,32>,  // threadblock
      cutlass::gemm::GemmShape<64,64,32>,    // warp
      cutlass::gemm::GemmShape<1,1,4>,       // instruction (DP4A)
      EpilogueOp>;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  typename Gemm::Arguments args(
      problem_size,
      {dA, lda},   // A col-major -> lda=M
      {dB, ldb},   // B col-major -> ldb=K
      {dC, ldc},   // C col-major -> ldc=M
      {dC, ldc},   // D = C (in-place)
      {alpha, beta}
  );

  Gemm gemm;

  // Workspace (CUTLASS 3.x/4.x)
  size_t workspace_bytes = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);

  return gemm(args, workspace.get(), stream);
}

int main() {
  // Problem
  int M = 1024, N = 1024, K = 64;

  // Host buffers
  std::vector<int8_t>  hA(M * K, 1);  // A filled with 1
  std::vector<int8_t>  hB(K * N, 2);  // B filled with 2
  std::vector<int32_t> hC(M * N, 0);  // C zeros

  // Device buffers
  int8_t  *dA = nullptr, *dB = nullptr;
  int32_t *dC = nullptr;
  cudaMalloc(&dA, sizeof(int8_t)  * M * K);
  cudaMalloc(&dB, sizeof(int8_t)  * K * N);
  cudaMalloc(&dC, sizeof(int32_t) * M * N);

  cudaMemcpy(dA, hA.data(), sizeof(int8_t)  * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), sizeof(int8_t)  * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), sizeof(int32_t) * M * N, cudaMemcpyHostToDevice);

  // Column-major leading dimensions
  int lda = M;  // A: rows
  int ldb = K;  // B: rows
  int ldc = M;  // C: rows

  // Alpha/Beta for integer epilogue
  int32_t alpha = 1;
  int32_t beta  = 0;

  // Warm-up (optional)
  matmul_int8(M, N, K, dA, lda, dB, ldb, dC, ldc, alpha, beta, 0);
  cudaDeviceSynchronize();

  // Time
  GpuTimer timer;
  timer.Start();
  cutlass::Status status = matmul_int8(M, N, K, dA, lda, dB, ldb, dC, ldc, alpha, beta, 0);
  timer.Stop();
  float time_ms = timer.Elapsed();
  std::cout << "CUTLASS IGEMM time: " << time_ms << " ms\n";

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS IGEMM failed: " << int(status) << "\n";
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return -1;
  }

  cudaMemcpy(hC.data(), dC, sizeof(int32_t) * M * N, cudaMemcpyDeviceToHost);

  // Since A=1, B=2 ⇒ each output = sum_k (1*2) = 2*K
  std::cout << "C[0]      = " << hC[0]        << " (expected " << 2 * K << ")\n";
  std::cout << "C[M*N-1]  = " << hC[M*N - 1]  << " (expected " << 2 * K << ")\n";

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
