#pragma once
#include <ttk/core/tensor.hpp>

namespace ttk {

template<typename T, int M, int D>
using Scalar = Tensor<T, M, D, 0>;

template<typename T, int M, int D>
using Vector = Tensor<T, M, D, 1>;

template<typename T, int M, int D>
using Tensor2 = Tensor<T, M, D, 2, false>;

template<typename T, int M, int D>
using SymmetricTensor2 = Tensor<T, M, D, 2, true>;

template<typename T, int M, int D>
using Tensor3 = Tensor<T, M, D, 3, false, false>;

template<typename T, int M, int D>
using Tensor4 = Tensor<T, M, D, 4, false, false, false>;

template<typename T, int M, int D>
using MajorSymmetricTensor4 = Tensor<T, M, D, 4, false, true, false>;

} // end namespace ttk