#pragma once

namespace ttk {
// ---------------- Backend detection ----------------

// CUDA: defined by nvcc
#if defined(__CUDACC__)
    #define TTK_CUDA 1
#else
    #define TTK_CUDA 0
#endif

// HIP: defined by hipcc
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    #define TTK_HIP 1
#else
    #define TTK_HIP 0
#endif

// macros needed by all backend backends
// TTK_FUNCTION
// TTK_INLINE_FUNCTION
// eventually TTK_FORCEINLINE_FUNCTION
// eventually kokkos like relocatable function

#if TENSOR_TOOLKIT_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

#define TTK_DEFAULTED_FUNCTION KOKKOS_DEFAULTED_FUNCTION
#define TTK_FUNCTION KOKKOS_FUNCTION
#define TTK_INLINE_FUNCTION KOKKOS_INLINE FUNCTION
#define TTK_FORCEINLINE_FUNCTION KOKKOS_FORCEINLINE_FUNCTION

#else

#define TTK_FUNCTION

#if TTK_CUDA | TTK_HIP

#define TTK_FUNCTION __host__ __device__
#define TTK_INLINE_FUNCTION __host__ __device__ inline

#else

// fallback do nothing
#define TTK_DEFAULTED_FUNCTION
#define TTK_FUNCTION
#define TTK_INLINE_FUNCTION
#define TTK_FORCEINLINE_FUNCTION

#endif

#endif

} // end namespace ttk
