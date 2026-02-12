#pragma once
#include "macros.hpp"
#include "tensor.hpp"

namespace ttk {

template<typename T>
TTK_FUNCTION
T __ifelse(const bool& flag, const T& a, const T& b) {
    if (flag) {
        return a;
    } else {
        return b;
    }
}

template<Scalar T>
TTK_FUNCTION
constexpr void __swap(T& a, T& b) noexcept {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

template<Scalar T, int M, int D>
TTK_FUNCTION
constexpr void __swap(Tensor<T, M, D, 2, FULL>& A, const int& i, const int& j) {
    for (int k = 0; k < D; ++k) {
        T temp = std::move(A(k, i));
        A(k, i) = std::move(A(k, j));
        A(k, j) = std::move(temp);
        // T temp = std::move(A(i, k));
        // A(i, k) = std::move(A(j, k));
        // A(j, k) = std::move(temp);
    }
}

} // end namespace ttk
