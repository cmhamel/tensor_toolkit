#pragma once
#include "macros.hpp"
#include "traits.hpp"
#if !defined(TENSOR_TOOLKIT_ENABLE_KOKKOS) || !TENSOR_TOOLKIT_ENABLE_KOKKOS
#include <random>
#endif

namespace ttk {

// NOTE this guy is likely not GPU safe
// but meant to mirror an eventual GPU implementation
// using Kokkos

#ifdef TENSOR_TOOLKIT_ENABLE_KOKKOS

#include <Kokkos_Core.hpp>

template<Scalar T>
struct UniformDistribution {
public:
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using PoolType = Kokkos::Random_XorShift64_Pool<ExecSpace>;

    PoolType randPool;

    TTK_DEFAULTED_FUNCTION
    explicit UniformDistribution(uint64_t seed)
        : randPool(seed) {}

    TTK_FUNCTION
    void free_state() {
        randPool.free_state();
    }

    TTK_FUNCTION
    auto get_state() {
        return randPool.get_state();
    }

    TTK_FUNCTION
    T rand() {
        auto gen = get_state();
        return gen.drand();
    }
};
// TODO no kokkos cuda/hip
#else

// #include <random>

template<Scalar T>
struct UniformDistribution {
public:
    TTK_DEFAULTED_FUNCTION
    explicit UniformDistribution()
        : dist(std::uniform_real_distribution<T>(0.0, 1.0)),
          rng(std::random_device{}()) {}

    // to mimic kokkos behavior
    TTK_FUNCTION
    T rand() {
        return dist(rng);
    }

    // to mimic kokkos behavior
    TTK_FUNCTION
    void free_state() {}; 
    // to mimic kokkos behavior
    TTK_FUNCTION
    void get_state() {};

    std::uniform_real_distribution<T> dist;
    std::mt19937 rng;
};

#endif

} // end namespace ttk

