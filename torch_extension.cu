#include <vector>
#include <algorithm>
#include <iostream>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/gemm/device/gemm_batched.h"

#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/arch/arch.h>

#include <cutlass/arch/mma.h>

#if __has_include(<cutlass/layout/matrix.h>)
#include <cutlass/layout/matrix.h>
#else
#include <cutlass/layout/row_major.h>
#include <cutlass/layout/column_major.h>
#endif

// -------------------------
// 1) DOT PRODUCT
// -------------------------
__global__ void dot_kernel(const float *__restrict__ a,
						   const float *__restrict__ b,
						   float *__restrict__ out,
						   long long N)
{
	float local = 0.0f;
	for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
		 i += (long long)blockDim.x * gridDim.x)
	{
		local += a[i] * b[i];
	}
	__shared__ float smem[256];
	int tid = threadIdx.x;
	smem[tid] = local;
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
			smem[tid] += smem[tid + s];
		__syncthreads();
	}
	if (tid == 0)
		atomicAdd(out, smem[0]);
}

torch::Tensor dot_forward(torch::Tensor a, torch::Tensor b)
{
	TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
	TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
				"float32 only (dot)");
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

// -------------------------
// 2) MATMUL FLOAT32: C = A @ B
//    A: (M,K), B: (K,N), C: (M,N)
// -------------------------
constexpr int TILE = 16;

__global__ void matmul_f32_kernel(const float *__restrict__ A,
								  const float *__restrict__ B,
								  float *__restrict__ C,
								  int M, int K, int N)
{
	__shared__ float As[TILE][TILE];
	__shared__ float Bs[TILE][TILE];

	int row = blockIdx.y * TILE + threadIdx.y; // [0..M)
	int col = blockIdx.x * TILE + threadIdx.x; // [0..N)

	float acc = 0.0f;
	// loop over tiles of K dimension
	for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
	{
		int a_col = t * TILE + threadIdx.x;
		int b_row = t * TILE + threadIdx.y;

		As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
		Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

		__syncthreads();

#pragma unroll
		for (int k = 0; k < TILE; ++k)
		{
			acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}
		__syncthreads();
	}

	if (row < M && col < N)
	{
		C[row * N + col] = acc;
	}
}

torch::Tensor matmul_f32_forward(torch::Tensor A, torch::Tensor B)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
	TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat, "float32 only");
	TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D (M,K) and (K,N)");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
	const int64_t M = A.size(0);
	const int64_t K = A.size(1);
	TORCH_CHECK(B.size(0) == K, "Inner dims must match: A(M,K) x B(K,N)");
	const int64_t N = B.size(1);

	auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat));
	dim3 block(TILE, TILE);
	dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

	auto stream = at::cuda::getCurrentCUDAStream();
	matmul_f32_kernel<<<grid, block, 0, stream.stream()>>>(
		A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
		(int)M, (int)K, (int)N);
	return C;
}

torch::Tensor matmul_f32_cutlass_forward(torch::Tensor A, torch::Tensor B)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
	TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat, "float32 only");
	TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A:(M,K), B:(K,N)");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contiguous required");

	int64_t M = A.size(0), K = A.size(1), N = B.size(1);
	TORCH_CHECK(B.size(0) == K, "A(M,K) x B(K,N) inner dim mismatch");

	auto C = torch::empty({M, N}, A.options());

	using RowMajor = cutlass::layout::RowMajor;
	using Gemm = cutlass::gemm::device::Gemm<
		float, RowMajor, // A
		float, RowMajor, // B
		float, RowMajor	 // C
		// Accumulator defaults to float; opclass/arch deduced from -arch
		>;

	Gemm gemm_op;
	Gemm::Arguments args(
		{int(M), int(N), int(K)},
		{A.data_ptr<float>(), int(K)}, // lda = K for row-major
		{B.data_ptr<float>(), int(N)}, // ldb = N
		{C.data_ptr<float>(), int(N)}, // ldc
		{C.data_ptr<float>(), int(N)},
		{1.0f, 0.0f} // alpha, beta
	);

	auto status = gemm_op(args, at::cuda::getCurrentCUDAStream());
	TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS f32 GEMM failed");
	return C;
}

// -------------------------
// 3) MATMUL INT8: C = A @ B
//    A: (M,K) int8, B: (K,N) int8, C: (M,N) int32 (accumulators)
//    (No quant/dequant here; just raw int8 matmul -> int32)
// -------------------------
__global__ void matmul_int8_kernel(const int8_t *__restrict__ A,
								   const int8_t *__restrict__ B,
								   int32_t *__restrict__ C,
								   int M, int K, int N)
{
	__shared__ int8_t As[TILE][TILE];
	__shared__ int8_t Bs[TILE][TILE];

	int row = blockIdx.y * TILE + threadIdx.y; // [0..M)
	int col = blockIdx.x * TILE + threadIdx.x; // [0..N)

	int acc = 0;
	for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
	{
		int a_col = t * TILE + threadIdx.x;
		int b_row = t * TILE + threadIdx.y;

		As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
		Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0;

		__syncthreads();

#pragma unroll
		for (int k = 0; k < TILE; ++k)
		{
			// promote to int
			acc += static_cast<int>(As[threadIdx.y][k]) * static_cast<int>(Bs[k][threadIdx.x]);
		}
		__syncthreads();
	}

	if (row < M && col < N)
	{
		C[row * N + col] = acc;
	}
}

torch::Tensor matmul_int8_forward(torch::Tensor A, torch::Tensor B)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
	TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar,
				"Inputs must be int8 (torch.int8)");
	TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D (M,K) and (K,N)");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");
	const int64_t M = A.size(0);
	const int64_t K = A.size(1);
	TORCH_CHECK(B.size(0) == K, "Inner dims must match: A(M,K) x B(K,N)");
	const int64_t N = B.size(1);

	auto C = torch::empty({M, N}, A.options().dtype(torch::kInt));
	dim3 block(TILE, TILE);
	dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

	auto stream = at::cuda::getCurrentCUDAStream();
	matmul_int8_kernel<<<grid, block, 0, stream.stream()>>>(
		reinterpret_cast<const int8_t *>(A.data_ptr<signed char>()),
		reinterpret_cast<const int8_t *>(B.data_ptr<signed char>()),
		C.data_ptr<int32_t>(),
		(int)M, (int)K, (int)N);
	return C;
}

torch::Tensor matmul_int8_cutlass_forward(torch::Tensor A, torch::Tensor B)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
	TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar, "int8 only");
	TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A:(M,K), B:(K,N)");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contiguous required");

	int64_t M = A.size(0), K = A.size(1), N = B.size(1);
	TORCH_CHECK(B.size(0) == K, "A(M,K) x B(K,N) inner dim mismatch");

	auto C = torch::empty({M, N}, A.options().dtype(torch::kInt));

	using RowMajor = cutlass::layout::RowMajor;

	// Simple int8->int32 GEMM (Tensor Cores on SM75+/SM80+ if arch flags set)
	using GemmInt8 = cutlass::gemm::device::Gemm<
		int8_t, RowMajor,  // A
		int8_t, RowMajor,  // B
		int32_t, RowMajor, // C
		int32_t			   // accumulator
		// You can also specify OpClassTensorOp, arch::Sm80, etc., if you want to lock it
		>;

	GemmInt8 gemm_op;
	GemmInt8::Arguments args(
		{int(M), int(N), int(K)},
		{A.data_ptr<int8_t>(), int(K)},	 // lda
		{B.data_ptr<int8_t>(), int(N)},	 // ldb
		{C.data_ptr<int32_t>(), int(N)}, // ldc
		{C.data_ptr<int32_t>(), int(N)},
		{1, 0} // alpha, beta (int)
	);

	auto status = gemm_op(args, at::cuda::getCurrentCUDAStream());
	TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS int8 GEMM failed");
	return C;
}

// -----------------------------------------------------------
// PyTorch-facing function:
//   D_fp16 = alpha_scale * (A_int8 @ B_int8)
//   A: [M,K] int8 contiguous, B: [K,N] int8 contiguous
//   Returns D: [M,N] float16
// -----------------------------------------------------------
torch::Tensor matmul_int8_to_fp16_scaled_forward_noc(
	const torch::Tensor &A,
	const torch::Tensor &B,
	double alpha_scale // scalar
)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
	TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar, "A,B must be int8");
	TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A:[M,K], B:[K,N]");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A,B must be contiguous");

	int64_t M = A.size(0), K = A.size(1), N = B.size(1);
	TORCH_CHECK(B.size(0) == K, "Inner dimension mismatch: A(M,K) x B(K,N)");

	// Output D is FP16 (row-major)
	auto D = torch::empty({M, N}, A.options().dtype(torch::kFloat16));

	// ---- CUTLASS types  ----
	using ElementA = int8_t;
	using ElementB = int8_t;
	using ElementAcc = int32_t;
	using ElementOut = cutlass::half_t;
	using ComputeEpi = float;
	using LayoutRM = cutlass::layout::RowMajor;

	static int const kElementsPerAccess = 1;
	using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
		ElementOut, kElementsPerAccess, ElementAcc, ComputeEpi>;

	using GemmSIMT = cutlass::gemm::device::Gemm<
		ElementA, LayoutRM,
		ElementB, LayoutRM,
		ElementOut, LayoutRM,
		ElementAcc,
		cutlass::arch::OpClassSimt,
		cutlass::arch::Sm80, // adjust arch if needed
		cutlass::gemm::GemmShape<128, 128, 32>,
		cutlass::gemm::GemmShape<64, 64, 32>,
		cutlass::gemm::GemmShape<1, 1, 4>,
		EpilogueOp>;

	cutlass::gemm::GemmCoord problem_size{int(M), int(N), int(K)};

	int lda = int(K), ldb = int(N), ldc = int(N), ldd = int(N);

	// No C/beta: pass D as C (valid pointer; beta=0 so it's not read)
	const ElementOut *c_ptr =
		reinterpret_cast<const ElementOut *>(D.data_ptr<at::Half>());

	typename GemmSIMT::Arguments args(
		problem_size,
		{A.data_ptr<ElementA>(), lda},
		{B.data_ptr<ElementB>(), ldb},
		{c_ptr, ldc},
		{reinterpret_cast<ElementOut *>(D.data_ptr<at::Half>()), ldd},
		{static_cast<float>(alpha_scale), 0.0f});

	GemmSIMT op;
	auto stream = at::cuda::getCurrentCUDAStream();

	// *** FIX 2: use cudaMalloc/free for workspace (no device_memory::allocation) ***
	size_t ws_bytes = GemmSIMT::get_workspace_size(args);
	void *workspace = nullptr;
	if (ws_bytes)
	{
		TORCH_CHECK(cudaMalloc(&workspace, ws_bytes) == cudaSuccess, "workspace cudaMalloc failed");
	}

	cutlass::Status status = op(args, workspace, stream);

	if (workspace)
		cudaFree(workspace);
	TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS fused int8->fp16 (SIMT) failed");

	return D;
}

// ==================================

// -------------------------
// 4) BATCHED MATMUL INT8: C = A @ B
//    A: (B,M,K) int8, B: (B,K,N) int8, C: (B,M,N) int32 (accumulators)
__global__ void bmatmul_int8_kernel(const int8_t *__restrict__ A,
									const int8_t *__restrict__ B,
									int32_t *__restrict__ C,
									int Bsz, int M, int K, int N)
{
	__shared__ int8_t As[TILE][TILE];
	__shared__ int8_t Bs[TILE][TILE];

	int b = blockIdx.z;						   // batch index
	int row = blockIdx.y * TILE + threadIdx.y; // [0..M)
	int col = blockIdx.x * TILE + threadIdx.x; // [0..N)

	// base offsets for this batch (row-major, contiguous)
	long long A_base = (long long)b * M * K;
	long long B_base = (long long)b * K * N;
	long long C_base = (long long)b * M * N;

	int acc = 0;
	for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
	{
		int a_col = t * TILE + threadIdx.x; // k for A
		int b_row = t * TILE + threadIdx.y; // k for B

		As[threadIdx.y][threadIdx.x] =
			(row < M && a_col < K) ? A[A_base + row * (long long)K + a_col] : 0;

		Bs[threadIdx.y][threadIdx.x] =
			(b_row < K && col < N) ? B[B_base + b_row * (long long)N + col] : 0;

		__syncthreads();

#pragma unroll
		for (int kk = 0; kk < TILE; ++kk)
		{
			acc += static_cast<int>(As[threadIdx.y][kk]) *
				   static_cast<int>(Bs[kk][threadIdx.x]);
		}
		__syncthreads();
	}

	if (row < M && col < N)
	{
		C[C_base + row * (long long)N + col] = acc;
	}
}

// --- Forward launcher: bmm_int8(A,B) -> C ---
torch::Tensor bmatmul_int8_forward(torch::Tensor A, torch::Tensor B)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
	TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar,
				"Inputs must be int8 (torch.int8)");
	TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "A and B must be 3D: (B,M,K) and (B,K,N)");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

	const int64_t Bsz = A.size(0);
	const int64_t M = A.size(1);
	const int64_t K = A.size(2);
	TORCH_CHECK(B.size(0) == Bsz && B.size(1) == K, "Batch and K must match");
	const int64_t N = B.size(2);

	auto C = torch::empty({Bsz, M, N}, A.options().dtype(torch::kInt));

	dim3 block(TILE, TILE, 1);
	dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, Bsz);

	auto stream = at::cuda::getCurrentCUDAStream();

	bmatmul_int8_kernel<<<grid, block, 0, stream.stream()>>>(
		reinterpret_cast<const int8_t *>(A.data_ptr<signed char>()),
		reinterpret_cast<const int8_t *>(B.data_ptr<signed char>()),
		C.data_ptr<int32_t>(),
		(int)Bsz, (int)M, (int)K, (int)N);

	return C;
}

// =================

torch::Tensor matmul_int8_cutlass_forward_on_stream(torch::Tensor A, torch::Tensor B, cudaStream_t stream)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
	TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar, "int8 only");
	TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A:(M,K), B:(K,N)");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contiguous required");

	int64_t M = A.size(0), K = A.size(1), N = B.size(1);
	TORCH_CHECK(B.size(0) == K, "A(M,K) x B(K,N) inner dim mismatch");

	auto C = torch::empty({M, N}, A.options().dtype(torch::kInt));

	using RowMajor = cutlass::layout::RowMajor;
	using GemmInt8 = cutlass::gemm::device::Gemm<
		int8_t, RowMajor,
		int8_t, RowMajor,
		int32_t, RowMajor,
		int32_t>;

	GemmInt8 gemm_op;
	GemmInt8::Arguments args(
		{int(M), int(N), int(K)},
		{A.data_ptr<int8_t>(), int(K)},	 // lda
		{B.data_ptr<int8_t>(), int(N)},	 // ldb
		{C.data_ptr<int32_t>(), int(N)}, // ldc
		{C.data_ptr<int32_t>(), int(N)},
		{1, 0} // alpha, beta (int)
	);

	auto status = gemm_op(args, stream); // <<< use the stream passed in
	TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS int8 GEMM (stream) failed");
	return C;
}

// ========== 3) Batched wrapper: call the stream version in parallel ==========
torch::Tensor bmm_int8_cutlass_forward_streams(torch::Tensor A, torch::Tensor B)
{
	TORCH_CHECK(A.is_cuda() && B.is_cuda(), "CUDA tensors required");
	TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar, "int8 only");
	TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contiguous required");
	TORCH_CHECK(A.dim() == 3, "A must be [B, M, K]");

	const int64_t BATCH = A.size(0);
	const int64_t M = A.size(1);
	const int64_t K = A.size(2);

	bool shared_B = false;
	int64_t N = 0;
	if (B.dim() == 2)
	{
		TORCH_CHECK(B.size(0) == K, "B(shared) must be [K, N]");
		N = B.size(1);
		shared_B = true;
	}
	else
	{
		TORCH_CHECK(B.dim() == 3, "B must be [K,N] or [B,K,N]");
		TORCH_CHECK(B.size(0) == BATCH && B.size(1) == K, "B is [B,K,N] and must match A");
		N = B.size(2);
	}

	// Output: int32 [B, M, N]
	auto C = torch::empty({BATCH, M, N}, A.options().dtype(torch::kInt));

	// Create a small pool of CUDA streams (simple and fast)
	const int device_index = A.get_device();
	const int num_streams = int(std::min<int64_t>(BATCH, 4)); // try 2–4 typically
	std::vector<at::cuda::CUDAStream> at_streams;
	at_streams.reserve(num_streams);
	for (int i = 0; i < num_streams; ++i)
	{
		at_streams.push_back(at::cuda::getStreamFromPool(/*high_priority=*/false, device_index));
	}

	// Enqueue each 2D matmul on a worker stream (no guards; we pass cudaStream_t directly)
	for (int64_t b = 0; b < BATCH; ++b)
	{
		at::cuda::CUDAStream s = at_streams[size_t(b % num_streams)];
		cudaStream_t raw = s.stream();

		torch::Tensor Ab = A.select(0, b).contiguous();				   // [M, K]
		torch::Tensor Bb = shared_B ? B : B.select(0, b).contiguous(); // [K, N]

		torch::Tensor Cb = matmul_int8_cutlass_forward_on_stream(Ab, Bb, raw); // [M, N] int32

		// Async copy into the output slice on the same stream
		C.select(0, b).copy_(Cb, /*non_blocking=*/true);
	}

	// Wait for all worker streams to finish
	for (auto &s : at_streams)
	{
		s.synchronize();
	}

	return C;
}

// -------------------------
// PYBIND
// -------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("dot_forward", &dot_forward, "Dot product (CUDA)");
	m.def("matmul_f32", &matmul_f32_forward, "Matmul float32: C=A@B (CUDA)");
	m.def("matmul_f32_cutlass", &matmul_f32_cutlass_forward, "CUTLASS GEMM float32");

	m.def("matmul_int8", &matmul_int8_forward, "Matmul int8->int32: C=A@B (CUDA)");
	m.def("matmul_int8_cutlass", &matmul_int8_cutlass_forward, "CUTLASS GEMM int8->int32");

	m.def("matmul_int8_to_fp16_scaled_forward_noc",
		  &matmul_int8_to_fp16_scaled_forward_noc,
		  "Fused int8 matmul -> fp16 with scale (beta=0, no C)");

	m.def("bmm_int8", &bmatmul_int8_forward, "Batched matmul int8->int32: C=A@B (CUDA)");
	m.def("bmm_int8_cutlass_forward_streams",
		  &bmm_int8_cutlass_forward_streams,
		  "Batched INT8 matmul via CUTLASS (parallel across CUDA streams)");
}
