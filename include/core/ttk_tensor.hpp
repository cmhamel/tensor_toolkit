#pragma once
#include <cassert> // for assert (TODO replace with better error handling)
#include <iomanip>
#include "ttk_indexing.hpp"
#include "ttk_lengths.hpp"
#include "ttk_macros.hpp"

namespace ttk {

template<typename T, int M, int D, int O, bool... Args>
struct Tensor {
public:
    // raw template arg helps
    static constexpr int Dimension = D;
    static constexpr int MetricType = M;
    static constexpr int Order = O;
    static constexpr bool SymArgs[sizeof...(Args)] = { Args... };
    using ValueType = T;

    // derived template args
    static constexpr int Length = TensorLength<D, O, Args...>::value;

    TTK_FUNCTION
    // TTK_DEFAULTED_FUNCTION
    Tensor() noexcept = default;
    // TTK_FUNCTION
    // Tensor() noexcept {
    //     fill(T{});
    // }

    TTK_FUNCTION
    Tensor(T value) noexcept {
        fill(value);
    }

    // TODO fix assert
    // TTK_FUNCTION
    // Tensor(std::initializer_list<T> data_) noexcept {
    //     assert(data_.size() == Length);
    //     int i = 0;
    //     for (T v : data_) {
    //         data[i++] = v;
    //     }
    // }
    TTK_FUNCTION
    constexpr Tensor(std::initializer_list<T> data_) noexcept {
        assert(data_.size() == Length);
        std::copy(data_.begin(), data_.end(), data);
    }

    TTK_FUNCTION
    void fill(T value) noexcept {
        std::fill(std::begin(data), std::end(data), value);
    }

    TTK_FUNCTION
    // constexpr T* getData() {
    T* getData() const {
        // return data;
        return data;
    }

    TTK_FUNCTION
    const T* getDataConst() const {
        return data;
    }

    TTK_FUNCTION
    static constexpr int getLength() {
        return Length;
    }

    template<typename... Ids>
    TTK_FUNCTION
    constexpr int linear_index(Ids... ids) const {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return LinearIndex<O, Args...>::compute(D, ids...);
    }

    // operator overloading
    template<typename... Ids>
    TTK_FUNCTION
    T operator()(Ids... ids) const {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data[linear_index(ids...)];
    }

    template<typename... Ids>
    TTK_FUNCTION
    T& operator()(Ids... ids) {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data[linear_index(ids...)];
    }

    TTK_FUNCTION
    Tensor operator+(const Tensor& B) const {
        Tensor C;
        for (int i = 0; i < Length; ++i) {
            C.data[i] = this->data[i] + B.data[i];
        }
        return C;
    }

    TTK_FUNCTION
    Tensor operator-(const Tensor& B) const {
        Tensor C;
        for (int i = 0; i < Length; ++i) {
            C.data[i] = this->data[i] - B.data[i];
        }
        return C;
    }

    TTK_FUNCTION
    Tensor operator*(const T& val) const {
        Tensor C;
        for (int i = 0; i < Length; ++i) {
            C.data[i] = val * this->data[i];
        }
        return C;
    }

private:
    T data[Length];
};

template<typename T, int M, int D>
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

template<typename T, int M, int D, bool Sym>
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

template<typename T, int M, int D, bool... Syms>
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

template<typename T, int M, int D, bool... Syms>
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
