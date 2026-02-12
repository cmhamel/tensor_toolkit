#pragma once
#include "macros.hpp"
#include "traits.hpp"

#ifdef TTK_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

using PoolType = Kokkos::Random_XorShift64_Pool<execution_space>;

// TODO restrict to double
// template<typename T>
//TODO actually implement for kokkos side of house

#else

#include <random>

namespace ttk {
// struct PoolType {
// public:
//     static thread_local std::mt19937 rng{std::random_device{}()};
// };

// NOTE this guy is likely not GPU safe
// but meant to mirror an eventual GPU implementation
// using Kokkos
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

} // end namespace ttk

#endif
