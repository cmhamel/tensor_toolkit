#pragma once
#include <cmath>
#include <core/ttk_globals.hpp>
#include <core/ttk_macros.hpp>
#include <core/ttk_tensor.hpp>
#include <ttk_aliases.hpp>

namespace ttk {

template<typename T>
TTK_INLINE
T cbrt(T base) {
    if constexpr (std::is_floating_point_v<T>) {
        #if defined(__CUDACC__) // device compilation
            return ::cbrt(base); // CUDA/HIP device pow
        #elif defined(__HIPCC__)
            return ::cbrt(base);
        #else
            return std::cbrt(base); // host
        #endif
    } else {
        // only supporting floats
        static_assert(
            __always_false<T>,
            "pow is only supported for floating point types"
        );
    }
}

// // cross
// template<typename T, int M>
// TTK_INLINE
// ContinuumVector<T, M, 3> cross(
//     const ContinuumVector<T, M, 3>& A,
//     const ContinuumVector<T, M, 3>& B
// ) {
//     return cross(A.getData(), B.getData());
// }

template<typename T, int M>
TTK_INLINE
Vector<T, M, 3> cross(
    const Vector<T, M, 3>& A,
    const Vector<T, M, 3>& B
) {
    Vector<T, M, 3> C;
    C(0) = A(1) * B(2) - A(2) * B(1);
    C(1) = A(2) * B(0) - A(0) * B(2);
    C(2) = A(0) * B(1) - A(1) * B(0);
    return C;
}

// // det
// template<typename T, int M, int D, bool Sym>
// TTK_INLINE
// T det(const ContinuumTensor2<T, M, D, Sym>& A) {
//     return det(A.getDataConst());
// }

template<typename T, int M, int D>
TTK_INLINE
T det(const Tensor2<T, M, D>& A) {
    if constexpr (D == 1) {
        return A(0, 0);
    }
    else if constexpr (D == 2) {
        return A(0, 0) * A(1, 1) -
               A(0, 1) * A(1, 0);
    }
    else if constexpr (D == 3) {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
               A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    }
    else {
        // Generic Laplace expansion (O(D!)) — OK for small D
        T detA = T(0);

        for (int j = 0; j < D; ++j) {
            Tensor<T, D-1, 2, false> minor;
            int r = 0;

            for (int i = 1; i < D; ++i) {
                int c = 0;
                for (int k = 0; k < D; ++k) {
                    if (k == j) continue;
                    minor(r, c++) = A(i, k);
                }
                ++r;
            }

            T sign = (j % 2 == 0) ? T(1) : T(-1);
            detA += sign * A(0, j) * det<T, D-1>(minor);
        }

        return detA;
    }
}

template<typename T, int M, int D>
TTK_INLINE
T det(const SymmetricTensor2<T, M, D>& A) {
    if constexpr (D == 1) {
        return A(0, 0);
    }
    else if constexpr (D == 2) {
        // | a b |
        // | b c |
        return A(0, 0) * A(1, 1) -
               A(0, 1) * A(0, 1);
    }
    else if constexpr (D == 3) {
        const T a = A(0, 0);
        const T b = A(1, 1);
        const T c = A(2, 2);

        const T d = A(0, 1);
        const T e = A(0, 2);
        const T f = A(1, 2);

        return
              a * (b * c - f * f)
            - d * (d * c - e * f)
            + e * (d * f - b * e);
    }
    else if constexpr (D == 4) {
        // Symmetric 4x4:
        // | a b c d |
        // | b e f g |
        // | c f h i |
        // | d g i j |

        const T a = A(0, 0);
        const T b = A(0, 1);
        const T c = A(0, 2);
        const T d = A(0, 3);

        const T e = A(1, 1);
        const T f = A(1, 2);
        const T g = A(1, 3);

        const T h = A(2, 2);
        const T i = A(2, 3);

        const T j = A(3, 3);

        return a * (e * (h * j - i * i) - f * (f * j - g * i) + g * (f * i - g * h)) -
               b * (b * (h * j - i * i) - f * (c * j - d * i) + g * (c * i - d * h)) +
               c * (b * (f * j - g * i) - e * (c * j - d * i) + g * (c * g - d * f)) -
               d * (b * (f * i - g * h) - e * (c * i - d * h) + f * (c * g - d * f));
    }
    // else {
    //     static_assert(
    //         D <= 4,
    //         "det(SymmetricTensor2): only implemented for D <= 4"
    //     );
    //     return T(0);
    // }
}

// dev
template<typename T, int M, int D, bool Sym>
TTK_INLINE
Tensor<T, M, D, 2, Sym> dev(const Tensor<T, M, D, 2, Sym>& A) {
    return A - (1. / 3.) * trace(A) * identity<T, M, D, 2, Sym>();
}

// template<typename T, int M>
// TTK_INLINE
// SpatialTensor2<T, M, 3> dott(const TwoPointTensor2<T, M, 3>& F) {
//     return dott(F.getDataConst());
// }

template<typename T, int M>
TTK_INLINE
SymmetricTensor2<T, M, 3> dott(const Tensor2<T, M, 3>& F) {
    SymmetricTensor2<T, M, 3> B;

    const T F00 = F(0, 0), F01 = F(0, 1), F02 = F(0, 2);
    const T F10 = F(1, 0), F11 = F(1, 1), F12 = F(1, 2);
    const T F20 = F(2, 0), F21 = F(2, 1), F22 = F(2, 2);

    // Diagonal
    B(0, 0) = F00 * F00 + F01 * F01 + F02 * F02;
    B(1, 1) = F10 * F10 + F11 * F11 + F12 * F12;
    B(2, 2) = F20 * F20 + F21 * F21 + F22 * F22;

    // Upper triangle
    B(0, 1) = F00 * F10 + F01 * F11 + F02 * F12;
    B(0, 2) = F00 * F20 + F01 * F21 + F02 * F22;
    B(1, 2) = F10 * F20 + F11 * F21 + F12 * F22;

    return B;
} 

template<typename T>
TTK_INLINE
T log(T base) {
    if constexpr (std::is_floating_point_v<T>) {
        #if defined(__CUDACC__) // device compilation
            return ::log(base); // CUDA/HIP device pow
        #elif defined(__HIPCC__)
            return ::log(base);
        #else
            return std::log(base); // host
        #endif
    } else {
        // fallback for integers (convert to double)
        // return pow(static_cast<double>(base), static_cast<double>(exp));
        static_assert(
            __always_false<T>,
            "pow is only supported for floating point types"
        );
    }
}

template<typename T, int M, int D>
TTK_INLINE
T norm(const Vector<T, M, D>& A) {
    T normA = 0.0;
    for (int i = 0; i < D; ++i) {
        normA = normA + A(i) * A(i);
    }
    return normA;
}

template<typename T, int M, int D>
TTK_INLINE
T norm(const Tensor2<T, M, D>& A) {
    T normA = 0.0;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            normA = normA + A(i, j) * A(i, j);
        }
    }
    return normA;
}

// TODO
// inv
// sqrt

template<typename T>
TTK_INLINE
T pow(T base, T exp) {
    if constexpr (std::is_floating_point_v<T>) {
        #if defined(__CUDACC__) // device compilation
            return ::pow(base, exp); // CUDA/HIP device pow
        #elif defined(__HIPCC__)
            return ::pow(base, exp);
        #else
            return std::pow(base, exp); // host
        #endif
    } else {
        // only supporting floats
        static_assert(
            __always_false<T>,
            "pow is only supported for floating point types"
        );
    }
}

template<typename T>
TTK_INLINE
T sqrt(T base) {
    if constexpr (std::is_floating_point_v<T>) {
        #if defined(__CUDACC__) // device compilation
            return ::sqrt(base); // CUDA/HIP device pow
        #elif defined(__HIPCC__)
            return ::sqrt(base);
        #else
            return std::sqrt(base); // host
        #endif
    } else {
        // fallback for integers (convert to double)
        // return pow(static_cast<double>(base), static_cast<double>(exp));
        static_assert(
            __always_false<T>,
            "pow is only supported for floating point types"
        );
    }
}

// trace
// TODO specialize for faster methods maybe?
// check what the compiler is doing with different
// levels of optimization
// template<typename T, int M, int D, bool Sym>
// TTK_INLINE
// T trace(const ContinuumTensor2<T, M, D, Sym>& A) {
//     return trace(A.getDataConst());
// }

template<typename T, int M, int D, bool Sym>
TTK_INLINE
T trace(const Tensor<T, M, D, 2, Sym>& A) {
    T trA = 0.0;
    for (int i = 0; i < D; ++i) {
        trA = trA + A(i, i);
    }
    return trA;
}

template<typename T, int M, int D>
TTK_INLINE
Tensor2<T, M, D> transpose(const Tensor2<T, M, D>& A) {
    Tensor2<T, M, D> At;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            At(i, j) = A(j, i);
        }
    }
    return At;
}

template<typename T, int M, int D>
TTK_INLINE
SymmetricTensor2<T, M, D> transpose(const SymmetricTensor2<T, M, D>& A) {
    return A;
}

// TODO need more methods for tranpose

// eigen
// eigvals
// eigvecs
// norm




// dotdot
// fromvoigt
// majorsymmetric
// majortranspose
// minorsymmetric
// minortranspose
// rotate
// rotationtensor
// skew
// symmetric
// tdot
template<typename T, int M>
TTK_INLINE
SymmetricTensor2<T, M, 3> tdot(const Tensor2<T, M, 3>& F) {
    SymmetricTensor2<T, M, 3> C;

    const T F00 = F(0, 0), F01 = F(0, 1), F02 = F(0, 2);
    const T F10 = F(1, 0), F11 = F(1, 1), F12 = F(1, 2);
    const T F20 = F(2, 0), F21 = F(2, 1), F22 = F(2, 2);

    // Diagonal
    C(0, 0) = F00 * F00 + F10 * F10 + F20 * F20;
    C(1, 1) = F01 * F01 + F11 * F11 + F21 * F21;
    C(2, 2) = F02 * F02 + F12 * F12 + F22 * F22;

    // Upper triangle
    C(0, 1) = F00 * F01 + F10 * F11 + F20 * F21;
    C(0, 2) = F00 * F02 + F10 * F12 + F20 * F22;
    C(1, 2) = F01 * F02 + F11 * F12 + F21 * F22;

    return C;
}
// tovoigt
// tovoigt!
template<typename T, int M, int D, bool Sym>
TTK_INLINE
T vol(const Tensor<T, M, D, 2, Sym>& A) {
    T trA = trace(A);
    return (1. / 3.) * trA;
}

} // end namespace ttk