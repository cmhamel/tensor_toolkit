#pragma once

#if TENSOR_TOOLKIT_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>

template <typename TensorType>
struct TensorTestFunctor {
    TensorType tensor;
    Kokkos::View<typename TensorType::ValueType*, Kokkos::DefaultExecutionSpace> out;

    TensorTestFunctor(TensorType t) : tensor(t), out("out", TensorType::Length) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        // simple test: copy tensor data to output
        auto data = tensor.getDataConst();
        if (i < TensorType::Length) {
            out(i) = data[i];
        }
    }

    auto get_view() const { return out; }
};

TEST(TTKUnitTests, TestKokkos) {
    Kokkos::ScopeGuard kokkosGuard; // initialize/teardown Kokkos
    // Create a tensor on the host
    ttk::Tensor2<double, ttk::CARTESIAN, 3> tens({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    // Wrap it in a functor
    TensorTestFunctor<decltype(tens)> functor(tens);

    // Launch a Kokkos parallel_for
    Kokkos::parallel_for("TestTensor", tens.Length, functor);

    std::cout << "Kokkos execution space: "
              << typeid(Kokkos::DefaultExecutionSpace).name()
              << "\n";
    std::cout << "Memory space: "
              << typeid(Kokkos::DefaultExecutionSpace::memory_space).name()
              << "\n";

    Kokkos::View<double*> d_view("d_view", 10);
    Kokkos::fence();

    // Copy back to host for checking
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), functor.get_view());

    // Check all elements
    for (int i = 0; i < tens.Length; ++i) {
        EXPECT_EQ(h_out(i), tens.getDataConst()[i]);
    }

    // simple kokkos test example
    Kokkos::View<int*, Kokkos::DefaultExecutionSpace> data("data", 1);

    Kokkos::parallel_for("test_device", 1, KOKKOS_LAMBDA(const int i){
        data(i) = 42;  // runs on device if backend is CUDA/HIP/OpenMP GPU
    });

    Kokkos::fence();  // ensure kernel completes

    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    EXPECT_EQ(h_data(0), 42);

}

#endif
