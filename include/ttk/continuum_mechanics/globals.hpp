#pragma once
#include <ttk/continuum_mechanics/continuum_tensor.hpp>
#include <type_traits>

namespace ttk {

template<typename T>
struct _TensorType;

// we don't need any vector traits luckily

template<typename T, int M, int D>
struct _TensorType<MaterialTensor2<T, M, D>> {
    static constexpr int value = 0;
};

template<typename T, int M, int D>
struct _TensorType<TwoPointTensor2<T, M, D>> {
    static constexpr int value = 1;
};

template<typename T, int M, int D>
struct _TensorType<TwoPointTensor2T<T, M, D>> {
    static constexpr int value = 2;
};

template<typename T, int M, int D>
struct _TensorType<SpatialTensor2<T, M, D>> {
    static constexpr int value = 3;
};

// TODO
// need traits for higher order tensors of different types

// concept to help with checking
// can safely operate together with 
// objectivity in mind
template<typename A, typename B>
concept IsObjective = 
    // if they're the same type, then we're ok
    std::is_same_v<A, B> ||
    // twopoint dot material
    (_TensorType<A>::value == 1 && _TensorType<B>::value == 0) ||
    // material dot twopoint tranpose
    (_TensorType<A>::value == 0 && _TensorType<B>::value == 2) ||
    // two point dot two point transpose
    (_TensorType<A>::value == 1 && _TensorType<B>::value == 2) ||
    // two point transpose dot spatial
    (_TensorType<A>::value == 2 && _TensorType<B>::value == 3) ||
    // spatial dot two point
    (_TensorType<A>::value == 3 && _TensorType<B>::value == 1);
    // TODO alot more cases to handle
// concept IsObjective = std::is_same_v<A, B>;

} // end namespace ttk
