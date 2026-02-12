#pragma once
#include <cmath>
#include <ttk/core/random.hpp>
#include <ttk/core/tensor.hpp>
#include <ttk/aliases.hpp>

namespace ttk {

template<typename TensorType>
constexpr TensorType identity() {
    using T = typename TensorType::ValueType;
    constexpr int M = TensorType::MetricType;
    constexpr int D = TensorType::Dimension;
    constexpr int O = TensorType::Order;
    constexpr size_t NumSymArgs = TensorType::NumSymArgs;
    constexpr auto SymArgs = TensorType::SymArgs;

    TensorType I;
    if constexpr (O == 1) {
        I.fill(1.0);
    } else if constexpr (O == 2) {
        I.fill(0.0);
        // if constexpr (D >= 1) I(0, 0) = T(1);
        // if constexpr (D >= 2) I(1, 1) = T(1);
        // if constexpr (D >= 3) I(2, 2) = T(1);
        // if constexpr (D >= 4) I(3, 3) = T(1);
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                if (i == j) {
                    I(i, j) = 1.0;
                }
            }
        }
    } else if constexpr (O == 4) {
        static_assert(NumSymArgs == 3, "Wrong number of symargs for tensor4");
        Tensor2<T, M, D> I2(identity<Tensor2<T, M, D>>());
        I.fill(0.0);
        if constexpr (!(SymArgs[0] && SymArgs[1] && SymArgs[2])) {
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    for (int k = 0; k < D; ++k) {
                        for (int l = 0; l < D; ++l) {
                            I(i, j, k, l) = I2(i, k) * I2(j, l);
                        }
                    }
                }
            }
        } else if constexpr (!(SymArgs[0] && !SymArgs[1] && SymArgs[2])) {
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    for (int k = 0; k < D; ++k) {
                        for (int l = 0; l < D; ++l) {
                            I(i, j, k, l) = 0.5 * (
                                I2(i, k) * I2(j, l) +
                                I2(i, l) * I2(j, k)
                            );
                        }
                    }
                }
            }
        } else {
            static_assert(__always_false<TensorType>, "Unsupported identity");
        }
    } else {
        static_assert(__always_false<TensorType>, "Unsupported identity");
    }
    return I;
}

template<typename T, int M, int D, int O, bool... Args>
TTK_FUNCTION
Tensor<T, M, D, O, Args...> ones() {
    Tensor<T, M, D, O, Args...> data;
    data.fill(1.0);
    return data;
}

template<typename T, int M>
TTK_FUNCTION
Tensor2<T, M, 3> random_orthogonal(UniformDistribution<T>& dist) {
    // Expect RNG to provide: double uniform01();
    // UniformDistribution<T> dist({});
    // T u1 = seed.uniform01();
    // T u2 = seed.uniform01();
    // T u3 = seed.uniform01();
    T u1 = dist.rand();
    T u2 = dist.rand();
    T u3 = dist.rand();

    const T two_pi = 6.283185307179586476925286766559;

    T s1 = sqrt(1.0 - u1);
    T s2 = sqrt(u1);

    T a1 = two_pi * u2;
    T a2 = two_pi * u3;

    T sin1 = sin(a1);
    T cos1 = cos(a1);
    T sin2 = sin(a2);
    T cos2 = cos(a2);

    // Unit quaternion
    T q1 = s1 * sin1;
    T q2 = s1 * cos1;
    T q3 = s2 * sin2;
    T q4 = s2 * cos2;

    Tensor2<T, M, 3> out;
    out.fill(0.0);

    out(0, 0) = 1 - 2 * (q2 * q2 + q3 * q3);
    out(0, 1) = 2 * (q1 * q2 - q3 * q4);
    out(0, 2) = 2 * (q1 * q3 + q2 * q4);

    out(1, 0) = 2 * (q1 * q2 + q3 * q4);
    out(1, 1) = 1 - 2 * (q1 * q1 + q3 * q3);
    out(1, 2) = 2 * (q2 * q3 - q1 * q4);

    out(2, 0) = 2 * (q1 * q3 - q2 * q4);
    out(2, 1) = 2 * (q2 * q3 + q1 * q4);
    out(2, 2) = 1 - 2 * (q1 * q1 + q2 * q2);

    return out;
}

template<typename TensorType>
constexpr TensorType uniform() {
    using T = typename TensorType::ValueType;
    UniformDistribution<T> dist;
    TensorType A;
    for (int i = 0; i < A.Length; ++i) {
        A.setData(i, dist.rand());
    }
    return A;
}

template<typename T, int M, int D, int O, bool... Args>
TTK_FUNCTION
Tensor<T, M, D, O, Args...> zero() {
    return Tensor<T, M, D, O, Args...>::zeros();
}

template<typename T, int M, int D, int O, bool... Args>
TTK_FUNCTION
Tensor<T, M, D, O, Args...> zeros() {
    Tensor<T, D, O, Args...> data;
    data.fill(0.0);
    return data;
}

// todo
// higher order identity tensors
// conversion methods from other types

} // end namespace ttk
