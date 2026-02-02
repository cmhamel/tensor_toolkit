#pragma once
#include <core/ttk_macros.hpp>
#include <core/ttk_tensor.hpp>
#include <ttk_aliases.hpp>

namespace ttk {

// TODO
// dot
template<typename T, int M>
TTK_INLINE
T dot(const Vector<T, M, 1> &A, const Vector<T, M, 1> &B) {
    return A(0) * B(0);
}

template<typename T, int M>
TTK_INLINE
T dot(const Vector<T, M, 2> &A, const Vector<T, M, 2> &B) {
    return A(0) * B(0) + A(1) * B(1);
}

template<typename T, int M>
TTK_INLINE
T dot(const Vector<T, M, 3> &A, const Vector<T, M, 3> &B) {
    return A(0) * B(0) + A(1) * B(1) + A(2) * B(2);
}

// TODO
// template<typename T>
// TTK_INLINE
// T dot(const Vector<T, 4> &A, const Vector<T, 4> &B) {
//     return A(0) * B(0) + A(1) * B(1) + A(2) * B(2) + A(3) * B(3);
// }

// TODO eventually specialize to symmetric vs. non-symmetric
// to trim a few ops
template<typename T, int M, int D, bool Sym>
TTK_INLINE
Vector<T, M, D> dot(const Tensor<T, M, D, 2, Sym> &A, const Vector<T, M, D> &B) {
    Vector<T, M, D> dotAB;
    for (int i = 0; i < D; ++i) {
        dotAB(i) = 0.0;
        for (int j = 0; j < D; ++j) {
            dotAB(i) = dotAB(i) + A(i, j) * B(j);
        }
    }
    return dotAB;
}

template<typename T, int M>
TTK_INLINE
Tensor<T, M, 3, 2, false> dot(
    const Tensor<T, 3, 2, false>& A,
    const Tensor<T, 3, 2, false>& B
) {
    Tensor<T, 3, 2, false> C;

    C(0, 0) = A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0) + A(0, 2) *B(2, 0);
    C(0, 1) = A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1) + A(0, 2) *B(2, 1);
    C(0, 2) = A(0, 0) * B(0, 2) + A(0, 1) * B(1, 2) + A(0, 2) *B(2, 2);

    C(1, 0) = A(1, 0) * B(0, 0) + A(1, 1) * B(1, 0) + A(1, 2) *B(2, 0);
    C(1, 1) = A(1, 0) * B(0, 1) + A(1, 1) * B(1, 1) + A(1, 2) *B(2, 1);
    C(1, 2) = A(1, 0) * B(0, 2) + A(1, 1) * B(1, 2) + A(1, 2) *B(2, 2);

    C(2, 0) = A(2, 0) * B(0, 0) + A(2, 1) * B(1, 0) + A(2, 2) *B(2, 0);
    C(2, 1) = A(2, 0) * B(0, 1) + A(2, 1) * B(1, 1) + A(2, 2) *B(2, 1);
    C(2, 2) = A(2, 0) * B(0, 2) + A(2, 1) * B(1, 2) + A(2, 2) *B(2, 2);

    return C;
}

// dcontract
// otimes
// otimesl
// otimesu

} // end namespace ttk