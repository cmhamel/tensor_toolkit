#pragma once
#include <cmath>
#include "helpers.hpp"
#include <limits>
#include "macros.hpp"
#include "tensor.hpp"
#include <ttk/aliases.hpp>

namespace ttk {

template<Scalar T>
TTK_FUNCTION
T abs(T val) {
    #if defined(__CUDACC__) // device compilation
        return ::abs(val); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::abs(val);
    #else
        return std::abs(val); // host
    #endif
}

template<Scalar T>
TTK_FUNCTION
T cbrt(T base) {
    #if defined(__CUDACC__) // device compilation
        return ::cbrt(base); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::cbrt(base);
    #else
        return std::cbrt(base); // host
    #endif
}

template<Scalar T, int M, int D>
TTK_FUNCTION
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

template<Scalar T, int M, int D>
TTK_FUNCTION
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
template<Scalar T, int M, int D, int Sym>
TTK_FUNCTION
Tensor<T, M, D, 2, Sym> dev(const Tensor<T, M, D, 2, Sym>& A) {
    return A - (1. / 3.) * trace(A) * identity<Tensor<T, M, D, 2, Sym>>();
}

template<Scalar T, int M, int D>
TTK_FUNCTION
SymmetricTensor2<T, M, 3> dott(const Tensor2<T, M, D>& F) {
    SymmetricTensor2<T, M, 3> B;
    dott(F, B);
    return B;
}

// in-place version
template<Scalar T, int M>
TTK_FUNCTION
void dott(
    const Tensor2<T, M, 3>& F,
    SymmetricTensor2<T, M, 3>& B
) {
    const T& F00 = F(0, 0), F01 = F(0, 1), F02 = F(0, 2);
    const T& F10 = F(1, 0), F11 = F(1, 1), F12 = F(1, 2);
    const T& F20 = F(2, 0), F21 = F(2, 1), F22 = F(2, 2);

    // voigt
    B.data[_ST2_00] = F00 * F00 + F01 * F01 + F02 * F02; // xx
    B.data[_ST2_11] = F10 * F10 + F11 * F11 + F12 * F12; // yy
    B.data[_ST2_22] = F20 * F20 + F21 * F21 + F22 * F22; // zz

    B.data[_ST2_01] = F00 * F10 + F01 * F11 + F02 * F12; // xy
    B.data[_ST2_02] = F00 * F20 + F01 * F21 + F02 * F22; // xz
    B.data[_ST2_12] = F10 * F20 + F11 * F21 + F12 * F22; // yz
} 

template<Scalar T>
TTK_FUNCTION
T exp(T base) {
    #if defined(__CUDACC__) // device compilation
        return ::exp(base); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::exp(base);
    #else
        return std::exp(base); // host
    #endif
}

template<Scalar T, int M, int D>
TTK_FUNCTION
SymmetricTensor2<T, M, D> exp(const SymmetricTensor2<T, M, D>& A) {
    auto [evals, evecs] = eigen(A);
    for (int i = 0; i < D; ++i) {
        evals(i) = exp(evals(i));
    }
    return from_eigen(evals, evecs);
}

template<Scalar T, int M>
TTK_FUNCTION
Tensor2<T, M, 3> inv(const Tensor2<T, M, 3>& A) {
    Tensor2<T, M, 3> Ainv;

    // shorthand
    const T& a00 = A(0, 0), a01 = A(0, 1), a02 = A(0, 2);
    const T& a10 = A(1, 0), a11 = A(1, 1), a12 = A(1, 2);
    const T& a20 = A(2, 0), a21 = A(2, 1), a22 = A(2, 2);

    // Compute cofactors
    const T c00 = a11 * a22 - a12 * a21;
    const T c01 = -(a10 * a22 - a12 * a20);
    const T c02 = a10 * a21 - a11 * a20;

    const T c10 = -(a01 * a22 - a02 * a21);
    const T c11 = a00 * a22 - a02 * a20;
    const T c12 = -(a00 * a21 - a01 * a20);

    const T c20 = a01 * a12 - a02 * a11;
    const T c21 = -(a00 * a12 - a02 * a10);
    const T c22 = a00 * a11 - a01 * a10;

    // Compute determinant
    const T detA = a00 * c00 + a01 * c01 + a02 * c02;

    // Check for singular matrix
    // Could add an assert or return zero tensor if det==0
    // For GPU, avoid exceptions; maybe just multiply by 1/det
    const T invDet = T(1) / detA;

    // Fill inverse
    Ainv(0, 0) = c00 * invDet;
    Ainv(0, 1) = c10 * invDet;
    Ainv(0, 2) = c20 * invDet;

    Ainv(1, 0) = c01 * invDet;
    Ainv(1, 1) = c11 * invDet;
    Ainv(1, 2) = c21 * invDet;

    Ainv(2, 0) = c02 * invDet;
    Ainv(2, 1) = c12 * invDet;
    Ainv(2, 2) = c22 * invDet;

    return inv;
}

template<Scalar T, int M>
TTK_FUNCTION
SymmetricTensor2<T, M, 3> inv(const SymmetricTensor2<T, M, 3>& A) {
    SymmetricTensor2<T, M, 3> Ainv;

    // shorthand
    const T& a00 = A(0, 0), a01 = A(0, 1), a02 = A(0, 2);
    const T&                a11 = A(1, 1), a12 = A(1, 2);
    const T&                               a22 = A(2, 2);

    // Compute cofactors
    const T c00 = a11 * a22 - a12 * a12;
    const T c01 = -(a01 * a22 - a12 * a02);
    const T c02 = a01 * a12 - a11 * a02;

    const T c11 = a00 * a22 - a02 * a02;
    const T c12 = -(a00 * a12 - a01 * a02);

    const T c22 = a00 * a11 - a01 * a01;

    // Compute determinant
    const T detA = a00 * c00 + a01 * c01 + a02 * c02;

    // Check for singular matrix
    // Could add an assert or return zero tensor if det==0
    // For GPU, avoid exceptions; maybe just multiply by 1/det
    const T invDet = T(1) / detA;

    // Fill inverse
    Ainv(0, 0) = c00 * invDet;
    Ainv(0, 1) = c01 * invDet;
    Ainv(0, 2) = c02 * invDet;

    Ainv(1, 1) = c11 * invDet;
    Ainv(1, 2) = c12 * invDet;

    Ainv(2, 2) = c22 * invDet;

    return inv;
}

template<Scalar T>
TTK_FUNCTION
T log(T base) {
    #if defined(__CUDACC__) // device compilation
        return ::log(base); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::log(base);
    #else
        return std::log(base); // host
    #endif
}

template<Scalar T, int M, int D>
TTK_FUNCTION
SymmetricTensor2<T, M, D> log(const SymmetricTensor2<T, M, D>& A) {
    auto [evals, evecs] = eigen(A);
    for (int i = 0; i < D; ++i) {
        evals(i) = log(evals(i));
    }
    return from_eigen(evals, evecs);
}

template<Scalar T>
TTK_FUNCTION
T max(T val1, T val2) {
    #if defined(__CUDACC__) // device compilation
        return ::max(val1, val2); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::max(val, val2);
    #else
        return std::max(val1, val2); // host
    #endif
}

template<Scalar T>
TTK_FUNCTION
T min(T val1, T val2) {
    #if defined(__CUDACC__) // device compilation
        return ::min(val1, val2); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::min(val1, val2);
    #else
        return std::min(val1, val2); // host
    #endif
}

template<Scalar T, int M, int D, int O, int... Syms>
TTK_FUNCTION
T norm_2(const Tensor<T, M, D, O, Syms...>& A) {
    return sqrt(norm_2_squared(A));
}

template<Scalar T, int M, int D>
TTK_FUNCTION
T norm_2_squared(const Vector<T, M, D>& A) {
    T normA = 0.0;
    for (int i = 0; i < D; ++i) {
        normA = normA + A(i) * A(i);
    }
    return normA;
}

template<Scalar T, int M, int D>
TTK_FUNCTION
T norm_2_squared(const Tensor2<T, M, D>& A) {
    T normA = 0.0;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            normA = normA + A(i, j) * A(i, j);
        }
    }
    return normA;
}

template<Scalar T, int M, int D, int O, int... Syms>
TTK_FUNCTION
T norm_inf(const Tensor<T, M, D, O, Syms...>& A) {
    T norm(-std::numeric_limits<T>::infinity());
    auto data = A.getDataConst();
    for (int i = 0; i < A.Length; ++i) {
        norm = max(norm, data[i]);
    }
    return norm;
}

// specialization for eigen vector reconstruction
template<Scalar T, int M, int D>
TTK_FUNCTION
SymmetricTensor2<T, M, D> otimes(const Vector<T, M, D>& a) {
    SymmetricTensor2<T, M, D> A;
    A(0, 0) = a(0) * a(0);
    A(1, 1) = a(1) * a(1);
    A(2, 2) = a(2) * a(2);
    A(0, 1) = a(0) * a(1);
    A(1, 2) = a(1) * a(2);
    A(2, 0) = a(2) * a(0);
    return A;
}

// TODO
// sqrt

template<Scalar T>
TTK_FUNCTION
T pow(T base, T exp) {
    #if defined(__CUDACC__) // device compilation
        return ::pow(base, exp); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::pow(base, exp);
    #else
        return std::pow(base, exp); // host
    #endif
}

template<Scalar T, int M, int D>
TTK_FUNCTION
SymmetricTensor2<T, M, D> pow(const SymmetricTensor2<T, M, D>& A, const T& m) {
    auto [evals, evecs] = eigen(A);
    for (int i = 0; i < D; ++i) {
        evals(i) = pow(evals(i), m);
    }
    return from_eigen(evals, evecs);
}

template<Scalar T>
TTK_FUNCTION
T sqrt(T base) {
    #if defined(__CUDACC__) // device compilation
        return ::sqrt(base); // CUDA/HIP device pow
    #elif defined(__HIPCC__)
        return ::sqrt(base);
    #else
        return std::sqrt(base); // host
    #endif
}

template<Scalar T, int M, int D>
TTK_FUNCTION
SymmetricTensor2<T, M, D> sqrt(const SymmetricTensor2<T, M, D>& A) {
    auto [evals, evecs] = eigen(A);
    for (int i = 0; i < D; ++i) {
        evals(i) = sqrt(evals(i));
    }
    return from_eigen(evals, evecs);
}

template<Scalar T>
TTK_FUNCTION
T sign(T val) {
    return __ifelse(val > T(0.0), T(1.0), T(-1.0));
}


// TODO need more methods for tranpose

// eigen
// eigvals
// eigvecs


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
// template<Scalar T, int M, int D>

template<Scalar T, int M, int D>
Tensor2<T, M, D> skew(const Tensor2<T, M, D>& A) {
    return 0.5 * (A - transpose(A));
}

template<Scalar T, int M, int D>
Tensor2<T, M, D> skew(const SymmetricTensor2<T, M, D>& A) {
    return zero<Tensor2<T, M, D>>();
}

template<Scalar T, int M, int D>
SymmetricTensor2<T, M, D> symmetric(const Tensor2<T, M, D>& A) {
    SymmetricTensor2<T, M, D> Asym;
    Asym(0, 0) = A(0, 0);
    Asym(1, 1) = A(1, 1);
    Asym(2, 2) = A(2, 2);
    Asym(0, 1) = 0.5 * (A(0, 1) + A(1, 0));
    Asym(1, 2) = 0.5 * (A(1, 2) + A(2, 1));
    Asym(2, 0) = 0.5 * (A(2, 0) + A(0, 2));
    return Asym;
}

template<Scalar T, int M, int D>
SymmetricTensor2<T, M, D> symmetric(const SymmetricTensor2<T, M, D>& A) {
    return A;
}

template<Scalar T, int M>
TTK_FUNCTION
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

// trace
// TODO specialize for faster methods maybe?
// check what the compiler is doing with different
// levels of optimization
template<Scalar T, int M, int D, int Sym>
TTK_FUNCTION
T trace(const Tensor<T, M, D, 2, Sym>& A) {
    // T trA = 0.0;
    // for (int i = 0; i < D; ++i) {
    //     trA = trA + A(i, i);
    // }
    // return trA;
    if constexpr (D == 1) {
        return A(0, 0);
    } else if constexpr (D == 2) {
        return A(0, 0) + A(1, 1);
    } else if constexpr (D == 3) {
        return A(0, 0) + A(1, 1) + A(2, 2);
    } else if constexpr (D == 4) {
        return A(0, 0) + A(1, 1) + A(2, 2) + A(3, 3);
    } else {
        static_assert(
            __always_false<T>,
            "Unsupported dimension in trace"
        );
    }
}

template<Scalar T, int M, int D>
TTK_FUNCTION
Tensor2<T, M, D> transpose(const Tensor2<T, M, D>& A) {
    Tensor2<T, M, D> At;
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            At(i, j) = A(j, i);
        }
    }
    return At;
}

template<Scalar T, int M, int D>
TTK_FUNCTION
SymmetricTensor2<T, M, D> transpose(const SymmetricTensor2<T, M, D>& A) {
    return A;
}

// tovoigt
// tovoigt!
template<Scalar T, int M, int D, int Sym>
TTK_FUNCTION
T vol(const Tensor<T, M, D, 2, Sym>& A) {
    T trA = trace(A);
    return (1. / 3.) * trA;
}

} // end namespace ttk