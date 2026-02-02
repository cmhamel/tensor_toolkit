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

#if defined(_MSC_VER)
    #define TTK_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define TTK_FORCE_INLINE __attribute__((always_inline)) inline
#else
    #define TTK_FORCE_INLINE inline
#endif

#if TTK_CUDA
    #define TTK_HOST_DEVICE __host__ __device__
#elif TTK_HIP
    #define TTK_HOST_DEVICE __host__ __device__
#else
    #define TTK_HOST_DEVICE
#endif

// developer facing macro
#define TTK_INLINE TTK_FORCE_INLINE TTK_HOST_DEVICE

} // end namespace ttk
