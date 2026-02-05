#pragma once
#include <core/ttk_globals.hpp>
#include <core/ttk_macros.hpp>
#include <core/ttk_tensor.hpp>
#include <ttk_aliases.hpp>
#include <ttk_continuum_tensor.hpp>

namespace ttk {

// cross
// template<typename T, int M>
// TTK_FUNCTION
// MaterialVector<T, M, 3> cross(
//     const MaterialVector<T, M, 3>& a,
//     const MaterialVector<T, M, 3>& b
// ) requires SameFrame<MaterialVector<T, M, 3>, MaterialVector<T, M, 3>> {
//     return cross(a.getDataConst(), b.getDataConst());
// }

// template<typename T, int M>
// TTK_FUNCTION
// SpatialVector<T, M, 3> cross(
//     const SpatialVector<T, M, 3>& a,
//     const SpatialVector<T, M, 3>& b
// ) requires SameFrame<SpatialVector<T, M, 3>, SpatialVector<T, M, 3>> {
//     return cross(a.getDataConst(), b.getDataConst());
// }

// template<typename A, typename B>
// TTK_FUNCTION
// auto cross(const A&, const B&) {
//     TTK_OBJECTIVITY_ERROR("cross", A, B);
// }

template<typename T, int M>
TTK_FUNCTION
Vector<T, M, 3> cross(
    const Vector<T, M, 3>& a,
    const Vector<T, M, 3>& b
) {
    Vector<T, M, 3> c;
    c(0) = a(1) * b(2) - a(2) * b(1);
    c(1) = a(2) * b(0) - a(0) * b(2);
    c(2) = a(0) * b(1) - a(1) * b(0);
    return c;
}

// TODO
// dot
// template<typename T, int M, int D>
// TTK_FUNCTION
// T dot(
//     const MaterialVector<T, M, D>& a,
//     const MaterialVector<T, M, D>& b
// ) requires SameFrame<MaterialVector<T, M, 3>, MaterialVector<T, M, 3>> {
//     return dot(a.getDataConst(), b.getDataConst());
// }

// template<typename T, int M, int D>
// TTK_FUNCTION
// T dot(
//     const SpatialVector<T, M, D>& a,
//     const SpatialVector<T, M, D>& b
// ) requires SameFrame<SpatialVector<T, M, 3>, SpatialVector<T, M, 3>> {
//     return dot(a.getDataConst(), b.getDataConst());
// }

// template<typename A, typename B>
// TTK_FUNCTION
// auto dot(const A&, const B&) {
//     TTK_OBJECTIVITY_ERROR("dot", A, B);
// }

template<typename T, int M, int D>
TTK_FUNCTION
T dot(const Vector<T, M, D>& a, const Vector<T, M, D>& b) {
    if constexpr (D == 0) {
        return a(0) * b(0);
    } else if constexpr (D == 1) {
        return a(0) * b(0) + a(1) * b(1);
    } else if constexpr (D == 2) {
        return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
    } else if constexpr (D == 3) {
        return a(0) * b(0) + a(1) * b(1) + a(2) * b(2) + a(3) * b(3);
    } else {
        static_assert(
            __always_false<T>(),
            "Dimension greater than 4 not supported"
        );
    }
}

// TODO eventually specialize to symmetric vs. non-symmetric
// to trim a few ops
template<typename T, int M, int D, bool Sym>
TTK_FUNCTION
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
TTK_FUNCTION
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