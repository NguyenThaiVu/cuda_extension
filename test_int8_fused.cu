// nvcc -O3 -std=c++17 -arch=sm_86 matmul_int8_to_fp16_scaled.cu -I/path/to/cutlass -o igemm_fp16
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/util/device_memory.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

// If CUTLASS 4.x consolidated layouts:
#if __has_include(<cutlass/layout/matrix.h>)
  #include <cutlass/layout/matrix.h>
#else
  #include <cutlass/layout/row_major.h>
  #include <cutlass/layout/column_major.h>
  #include <cutlass/layout/column_major_interleaved.h>
#endif

// ----------------------------------
// Simple GPU timer
struct GpuTimer {
  cudaEvent_t start{}, stop{};
  cudaStream_t stream{0};
  explicit GpuTimer(cudaStream_t s = 0) : stream(s) {
    cudaEventCreate(&start); cudaEventCreate(&stop);
  }
  ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  void Start() { cudaEventRecord(start, stream); }
  void Stop()  { cudaEventRecord(stop, stream); }
  float Elapsed() { cudaEventSynchronize(stop); float ms=0.f; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

// ----------------------------------
// Set 1 to use INT8 Tensor Cores (SM75+); 0 keeps SIMT/DP4A
#ifndef USE_TENSOR_OP
#define USE_TENSOR_OP 0
#endif

// ----------------------------------
// INT8 x INT8 -> (accum s32) -> FP16 in epilogue with scale
// D_fp16 = alpha * (A_int8 @ B_int8) + beta * C_fp16
// All matrices are column-major; lda=M, ldb=K, ldc=M, ldd=M
// ----------------------------------
cutlass::Status matmul_int8_to_fp16_scaled(
    int M, int N, int K,
    const int8_t* dA, int lda,
    const int8_t* dB, int ldb,
    cutlass::half_t* dD, int ldd,     // FP16 output
    float alpha_scale,                 // scale applied in epilogue
    const cutlass::half_t* dC = nullptr, int ldc = 0, // optional FP16 source C
    float beta = 0.0f,
    cudaStream_t stream = 0) {

  if (K % 4 != 0) {
    std::cerr << "[warn] K=" << K << " not multiple of 4; DP4A/TensorOp paths prefer multiples.\n";
  }

  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutCD = cutlass::layout::ColumnMajor;

  using ElementA   = int8_t;
  using ElementB   = int8_t;
  using ElementAccumulator = int32_t;       // s8*s8 accumulates in s32
  using ElementOutput      = cutlass::half_t; // (also used for source C)
  using ElementComputeEpilogue = float;     // alpha, beta applied in float

  // For SIMT epilogue we keep scalar access to be safe across shapes
  static int const kElementsPerAccess = 1;

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,           // Output element type (D and C)
      kElementsPerAccess,      // Elements per vectorized access
      ElementAccumulator,      // Accumulator (int32)
      ElementComputeEpilogue   // Compute for alpha/beta (float)
  >;

#if USE_TENSOR_OP
  // Tensor Core INT8 path (mma.sync.m8n8k32.s32.s8.s8.s32); adjust arch as needed
  using Gemm = cutlass::gemm::device::Gemm<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementOutput, LayoutCD,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm86,                   // Ampere+
      cutlass::gemm::GemmShape<128,128,64>,  // Threadblock
      cutlass::gemm::GemmShape<64,64,64>,    // Warp
      cutlass::gemm::GemmShape<8,8,32>,      // Instruction (INT8 TC)
      EpilogueOp>;
#else
  // SIMT / DP4A path (portable, compiles everywhere)
  using Gemm = cutlass::gemm::device::Gemm<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementOutput, LayoutCD,
      ElementAccumulator,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm86,                   // Set your target arch
      cutlass::gemm::GemmShape<128,128,32>,
      cutlass::gemm::GemmShape<64,64,32>,
      cutlass::gemm::GemmShape<1,1,4>,
      EpilogueOp>;
#endif

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // If no source C provided and beta==0, we can legally pass D as C (valid pointer).
  const cutlass::half_t* c_ptr = dC ? dC : dD;
  int ldc_eff = dC ? ldc : ldd;

  typename Gemm::Arguments args(
      problem_size,
      {dA, lda},
      {dB, ldb},
      {c_ptr, ldc_eff}, // Source C (beta may be 0)
      {dD, ldd},        // Destination D (FP16)
      {alpha_scale, beta}
  );

  Gemm gemm;

  // CUTLASS 3.x/4.x: workspace query + run
  size_t workspace_bytes = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);

  return gemm(args, workspace.get(), stream);
}

int main() {
  // Problem sizes
  int M = 1024, N = 1024, K = 64;

  // Column-major leading dims
  int lda = M;  // A rows
  int ldb = K;  // B rows
  int ldd = M;  // D rows
  int ldc = M;  // C rows (if used)

  // Host buffers
  std::vector<int8_t>          hA(M * K, 1);  // all ones
  std::vector<int8_t>          hB(K * N, 2);  // all twos
  std::vector<cutlass::half_t> hD(M * N);     // FP16 output
  std::vector<cutlass::half_t> hC(M * N, cutlass::half_t(0)); // optional FP16 source C

  // Device buffers
  int8_t *dA=nullptr, *dB=nullptr;
  cutlass::half_t *dD=nullptr, *dC=nullptr;

  cudaMalloc(&dA, sizeof(int8_t) * M * K);
  cudaMalloc(&dB, sizeof(int8_t) * K * N);
  cudaMalloc(&dD, sizeof(cutlass::half_t) * M * N);
  cudaMalloc(&dC, sizeof(cutlass::half_t) * M * N);

  cudaMemcpy(dA, hA.data(), sizeof(int8_t) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), sizeof(int8_t) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), sizeof(cutlass::half_t) * M * N, cudaMemcpyHostToDevice);

  // Scale to apply when converting int32 -> fp16 in epilogue
  // For A=1, B=2 => accumulation is (2*K). If you quantized with a scale S,
  // set alpha_scale=S (or 1/S, depending on your convention).
  float alpha_scale = 1.0f; // change to your dequant scale
  float beta = 0.0f;        // set >0 if you want to accumulate into existing FP16 C

  // Warmup
  (void)matmul_int8_to_fp16_scaled(M, N, K, dA, lda, dB, ldb,
                                   dD, ldd, alpha_scale, dC, ldc, beta, 0);
  cudaDeviceSynchronize();

  // Time a run
  GpuTimer timer;
  timer.Start();
  cutlass::Status status = matmul_int8_to_fp16_scaled(M, N, K, dA, lda, dB, ldb,
                                                      dD, ldd, alpha_scale, dC, ldc, beta, 0);
  timer.Stop();
  std::cout << "Fused INT8 GEMM -> FP16 (scaled) time: " << timer.Elapsed() << " ms\n";

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS fused GEMM failed: " << int(status) << "\n";
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    return -1;
  }

  // Copy back and check a couple of values
  cudaMemcpy(hD.data(), dD, sizeof(cutlass::half_t) * M * N, cudaMemcpyDeviceToHost);

  // Expected: (A=1, B=2) => sum_k(1*2) = 2*K; then scaled by alpha_scale
  float expected = (2.0f * K) * alpha_scale;
  auto h2f = [](cutlass::half_t h) { return static_cast<float>(h); };

  std::cout << "D[0]     = " << h2f(hD[0])       << " (expected ~" << expected << ")\n";
  std::cout << "D[last]  = " << h2f(hD.back())   << " (expected ~" << expected << ")\n";

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return 0;
}
