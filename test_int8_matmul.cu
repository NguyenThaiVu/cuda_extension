#include <iostream>

// CUDA Runtime
#include <cuda_runtime.h>

// CUTE headers, a required dependency
#include "cute/tensor.hpp"

// CUTLASS 4.x Core Headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

// Utility headers
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

// Macro to check for CUDA errors
#define CUDA_CHECK(status)                                                      \
  {                                                                             \
    cudaError_t error = status;                                                 \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line "   \
                << __LINE__ << std::endl;                                       \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

// Main GEMM execution function
void run_gemm() {
    // Define the problem size
    int M = 512;
    int N = 1024;
    int K = 256;

    std::cout << "Running GEMM for problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using ElementD = int32_t;
    using ElementAccumulator = int32_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using ArchTag = cutlass::arch::Sm80;

    // <<<<<<<<<<<< START: FIX #1 (Remove cute::Domain<>) >>>>>>>>>>>>
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 8,
        ElementB, LayoutB, 8,
        ElementAccumulator,
        cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>,
        cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>,
        cutlass::gemm::collective::StageCount<2>
        // cute::Domain<> was here and has been removed
    >::CollectiveOp;

    // <<<<<<<<<<<< START: FIX #2 (Simplify DefaultEpilogue) >>>>>>>>>>>>
    using CollectiveEpilogue = typename cutlass::epilogue::collective::DefaultEpilogue<
        CollectiveMainloop,
        cutlass::epilogue::thread::LinearCombination<ElementD, 1, ElementAccumulator, ElementAccumulator>
        // Extra template arguments removed
    >::Epilogue;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        CollectiveMainloop,
        CollectiveEpilogue,
        LayoutC
    >;
    // <<<<<<<<<<<< END: FIX #1 & #2 >>>>>>>>>>>>

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    cutlass::HostTensor<ElementA, LayoutA> tensor_A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> tensor_B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_C({M, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_ref_D({M, N});

    // <<<<<<<<<<<< START: FIX #3 (Correct initialization syntax) >>>>>>>>>>>>
    uint64_t seed = 42;
    cutlass::Distribution dist;
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_A.host_view(), seed, dist.uniform<int8_t>(-10, 10));
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_B.host_view(), seed + 1, dist.uniform<int8_t>(-10, 10));
    // <<<<<<<<<<<< END: FIX #3 >>>>>>>>>>>>

    cutlass::reference::host::TensorFill(tensor_C.host_view(), 0);
    cutlass::reference::host::TensorFill(tensor_ref_D.host_view(), 0);

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();

    ElementAccumulator alpha = 1;
    ElementAccumulator beta = 0;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {tensor_A.device_data(), tensor_A.stride(0)},
        {tensor_B.device_data(), tensor_B.stride(0)},
        {{alpha, beta}, tensor_C.device_data(), tensor_C.stride(0), tensor_C.device_data(), tensor_C.stride(0)}
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS kernel cannot implement this problem size. Error: " << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS GEMM kernel. Error: " << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run CUTLASS GEMM kernel. Error: " << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    tensor_C.sync_host();

    std::cout << "CUTLASS GEMM computation complete." << std::endl;

    std::cout << "Verifying result against CPU reference..." << std::endl;
    cutlass::reference::host::Gemm(
        {M, N, K},
        alpha,
        tensor_A.host_view(),
        tensor_B.host_view(),
        beta,
        tensor_ref_D.host_view(),
        tensor_ref_D.host_view()
    );

    bool passed = cutlass::reference::host::TensorEquals(tensor_C.host_view(), tensor_ref_D.host_view());
    if (passed) {
        std::cout << "✅ Verification Passed!" << std::endl;
    } else {
        std::cout << "❌ Verification FAILED!" << std::endl;
    }
}

int main() {
    run_gemm();
    return 0;
}