#pragma once
#include <core/ttk_globals.hpp>
#include <core/ttk_tensor.hpp>

namespace ttk {

template<typename T, int M, int D, int O, bool... Args>
TTK_FUNCTION
Tensor<T, M, D, O, Args...> fill(T val) {
    Tensor<T, M, D, O, Args...> data;
    data.fill(val);
    return data;
}

// template<typename TType>
// TType identity() {
//     // TType::TensorType I = __identity<TType::TensorType>();
//     using TensorType = typename TType::TensorType;
//     return __identity<TensorType>();
//     // return I;
// }

template<typename TensorType>
TensorType identity();

template<typename TensorType>
TensorType identity() {
    using T = typename TensorType::ValueType;
    constexpr int M = TensorType::MetricType;
    constexpr int O = TensorType::Order;
    constexpr int D = TensorType::Dimension;
    constexpr auto SymArgs = TensorType::SymArgs;

    TensorType I;
    if constexpr (O == 1) {
        I.fill(1.0);
    } else if constexpr (O == 2) {
        I.fill(0.0);
        if constexpr (D >= 1) I(0, 0) = T(1);
        if constexpr (D >= 2) I(1, 1) = T(1);
        if constexpr (D >= 3) I(2, 2) = T(1);
        if constexpr (D >= 4) I(3, 3) = T(1);
    } else {
        static_assert(__always_false<TensorType>, "Unsupported identity");
    }
    return I;
}

// template<typename TType>
// TType identity() {
//     // TType::TensorType I = __identity<TType::TensorType>();
//     using TensorType = typename TType::TensorType;
//     return __identity<TensorType>();
//     // return I;
// }

template<typename T, int M, int D, int O, bool... Args>
TTK_FUNCTION
Tensor<T, M, D, O, Args...> ones() {
    Tensor<T, M, D, O, Args...> data;
    data.fill(1.0);
    return data;
}

// template<typename T, int M, int D, int O, bool... Args>
// TTK_FUNCTION
// Tensor<T, M, D, O, Args...> tensor(std::initializer_list<T> data_) {
//     Tensor<T, M, D, O, Args...> A;
//     assert(data_.size() == Length);
//     std::copy(data_.begin(), data_.end(), A.data);
// }

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
