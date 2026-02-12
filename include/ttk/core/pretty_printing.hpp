#pragma once
#include "tensor.hpp"

namespace ttk {

template<Scalar T, int M, int D>
std::ostream& operator<<(std::ostream& os, const Tensor<T, M, D, 1>& A) {
    os << "Vector(:) = [\n";
    for (int i = 0; i < D; ++i) {
        os << std::setw(10) << std::fixed << std::setprecision(4) << A(i);
        if (i != D - 1) os << "\n";
    }
    os << "\n";
    os << "]";
    return os;
}

template<Scalar T, int M, int D, int Sym>
std::ostream& operator<<(std::ostream& os, const Tensor<T, M, D, 2, Sym>& A) {
    os << "Tensor(:, :) = [\n";
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            os << std::setw(10) << std::fixed << std::setprecision(4) << A(i, j);
        }
        if (i != D - 1) os << "\n";
    }
    os << "\n";
    os << "]";
    return os;
}

template<Scalar T, int M, int D, int... Syms>
std::ostream& operator<<(std::ostream& os, const Tensor<T, M, D, 3, Syms...>& A) {
    for (int i = 0; i < D; ++i) {
        os << "Tensor(" << i << ", " << ":, :) = [\n";
        for (int j = 0; j < D; ++j) {
            for (int k = 0; k < D; ++k) {
                os << std::setw(10) << std::fixed << std::setprecision(4) << A(i, j, k);
            }
            if (j != D - 1) os << "\n";
        }
        os << "\n";
        os << "]\n";
    }
    return os;
}

template<Scalar T, int M, int D, int... Syms>
std::ostream& operator<<(std::ostream& os, const Tensor<T, M, D, 4, Syms...>& A) {
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            os << "Tensor(" << i << ", " << j << ", :, :) = [\n";
            for (int k = 0; k < D; ++k) {
                os << "  ";
                for (int l = 0; l < D; ++l) {
                    os << std::setw(10) << std::fixed << std::setprecision(4)
                       << A(i, j, k, l);
                    if (l != D - 1) os << ", ";
                }
                os << "\n";
            }
            os << "]\n\n"; // end of slice + extra newline
        }
    }
    return os;
}

} // end namespace ttk
