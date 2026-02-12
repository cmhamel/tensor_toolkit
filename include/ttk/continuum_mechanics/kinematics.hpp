#pragma once
#include <ttk/aliases.hpp>
#include <ttk/core/enums.hpp>
#include <ttk/core/macros.hpp>

namespace ttk {

template<Scalar T, int M>
TTK_FUNCTION
T first_invariant(const Tensor2<T, M, 3>& F) {
    const T& F00 = F(0, 0), F01 = F(0, 1), F02 = F(0, 2);
    const T& F10 = F(1, 0), F11 = F(1, 1), F12 = F(1, 2);
    const T& F20 = F(2, 0), F21 = F(2, 1), F22 = F(2, 2);
    T ret = F00 * F00 + F01 * F01 + F02 * F02 + // 00
            F10 * F10 + F11 * F11 + F12 * F12 + // 11
            F20 * F20 + F21 * F21 + F22 * F22;  // 22
    return ret;
}

template<Scalar T, int M>
TTK_FUNCTION
T first_invariant(const SymmetricTensor2<T, M, 3>& C) {
    const T& C00 = C(0, 0), C01 = C(0, 1), C02 = C(0, 2);
    const T&                C11 = C(1, 1), C12 = C(1, 2);
    const T&                               C22 = C(2, 2);
    T ret = C00 * C00 + C01 * C01 + C02 * C02 + // 00
            C01 * C01 + C11 * C11 + C12 * C12 + // 11
            C02 * C02 + C12 * C12 + C22 * C22;  // 22
    return ret;
}

} // end namespace ttk
