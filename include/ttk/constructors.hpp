#pragma once
#include <ttk/core/globals.hpp>
#include <ttk/core/tensor.hpp>

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
// random tensors
// higher order identity tensors
// conversion methods from other types

} // end namespace ttk
