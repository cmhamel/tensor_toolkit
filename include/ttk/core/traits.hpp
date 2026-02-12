#pragma once
#include "enums.hpp"
#include "macros.hpp"

namespace ttk {

// helper to make things build correctly
// with certain compiler flags
template<typename...>
// TTK_FUNCTION
constexpr bool __always_false = false;

template<typename T>
concept Scalar = std::is_arithmetic_v<T>;

namespace _indexing {

TTK_FUNCTION
constexpr int pair_index(int D, int a, int b) noexcept {
    return a * D + b;   // no minor symmetry
}

TTK_FUNCTION
constexpr int upper_triangular(int i, int j) noexcept {
    return (i <= j)
        ? (j * (j + 1)) / 2 + i
        : (i * (i + 1)) / 2 + j;
}

template<int O, int... Syms>
struct LinearIndex;

template<>
struct LinearIndex<1> {
    template<typename... Ids>
    TTK_FUNCTION
    static constexpr int compute(int /*D*/, Ids... ids) noexcept {
        static_assert(sizeof...(Ids) == 1);
        return (ids, ...);
    }
};

template<>
struct LinearIndex<2, FULL> {
    TTK_FUNCTION
    static constexpr int compute(int D, int i, int j) noexcept {
        return i * D + j;
    }
};

template<>
struct LinearIndex<2, SYMM> {
    TTK_FUNCTION
    static constexpr int compute(const int D, int i, int j) noexcept {
        // canonicalize indices
        if (i > j) {
            int tmp = i;
            i = j;
            j = tmp;
        }

        return i * D
             - (i * (i - 1)) / 2
             + (j - i);
    }
};

template<>
struct LinearIndex<3, FULL, FULL> {
    TTK_FUNCTION
    static constexpr int compute(int D, int i, int j, int k) noexcept {
        return i * D * D + j * D + k;
    }
};

template<>
struct LinearIndex<4, FULL, FULL, FULL> {
    TTK_FUNCTION
    static constexpr int compute(int D, int i, int j, int k, int l) noexcept {
        return ((i * D + j) * D + k) * D + l;
    }
};

template<>
struct LinearIndex<4, FULL, SYMM, FULL> {

    TTK_FUNCTION
    static constexpr int compute(int D, int i, int j, int k, int l) noexcept {
        const int p = pair_index(D, i, j);
        const int q = pair_index(D, k, l);

        return upper_triangular(p, q);
    }
};

} // end namespace ttk::_indexing

namespace _lengths {

template<int D, int O, int... Syms>
struct _TensorLength;

// 1st order
template<int D>
struct _TensorLength<D, 1> {
    static constexpr int value = D;
};

// 2nd order
template<int D>
struct _TensorLength<D, 2, FULL> {
    static constexpr int value = D * D;
};

template<int D>
struct _TensorLength<D, 2, SYMM> {
    static constexpr int value = D * (D + 1) / 2;
};

// 3rd order
template<int D>
struct _TensorLength<D, 3, FULL, FULL> {
    static constexpr int value = D * D * D;
};

// 4th order
template<int D>
struct _TensorLength<D, 4, FULL, FULL, FULL> {
    static constexpr int value = D * D * D * D;
};

template<int D>
struct _TensorLength<D, 4, FULL, SYMM, FULL> {
    static constexpr int value = D * D * (D + 1) * (D + 1) / 2 / 2;
};

} // end namespace ttk::_lengths

template<int D, int O, int... Syms>
struct TensorLength {
    // TODO add more compile time checks that e.g.
    // order/dimension or sensible/supported
    // sanity check: number of symmetry flags must be order-1
    static_assert(
        sizeof...(Syms) == (O > 0 ? O - 1 : 0),
        "Number of symmetry flags must be O-1"
    );
    // static constexpr int value =
    //     (O == 0) ? 1 :               // scalar
    //     (O == 1) ? D :               // vector
    //     _TensorLength<D, Syms...>::value;
    static constexpr int value = 
        (O == 0) ? 1 :
        _lengths::_TensorLength<D, O, Syms...>::value;
};

} // end namespace ttk
