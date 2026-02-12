#pragma once
#include <ttk/core/tensor.hpp>

namespace ttk {

template<Scalar T, int M, int D>
using Vector = Tensor<T, M, D, 1>;

template<Scalar T, int M, int D>
using Tensor2 = Tensor<T, M, D, 2, FULL>;

template<Scalar T, int M, int D>
using SymmetricTensor2 = Tensor<T, M, D, 2, SYMM>;

template<Scalar T, int M, int D>
using Tensor3 = Tensor<T, M, D, 3, FULL, FULL>;

template<Scalar T, int M, int D>
using Tensor4 = Tensor<T, M, D, 4, FULL, FULL, FULL>;

template<Scalar T, int M, int D>
using MajorSymmetricTensor4 = Tensor<T, M, D, 4, FULL, SYMM, FULL>;

} // end namespace ttk
