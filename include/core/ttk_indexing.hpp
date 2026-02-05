#pragma once
#include "ttk_enums.hpp"
#include "ttk_macros.hpp"

namespace ttk {

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

template<int O, bool... Syms>
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
    static constexpr int compute(const int D, const int i, const int j) noexcept {
        // canonicalize indices
        // if (i > j) {
        //     int tmp = i;
        //     i = j;
        //     j = tmp;
        // }

        // return i * D
        //      - (i * (i - 1)) / 2
        //      + (j - i);
        if (i == 0) {
            if (j == 0) {
                return _ST2_00;
            } else if (j == 1) {
                return _ST2_01;
            } else if (j == 2) {
                return _ST2_02;
            } else {
                assert(false);
            }
        } else if (i == 1) {
            if (j == 0) {
                return _ST2_10;
            } else if (j == 1) {
                return _ST2_11;
            } else if (j == 2) {
                return _ST2_12;
            } else {
                assert(false);
            }
        } else if (i == 2) {
            if (j == 0) {
                return _ST2_20;
            } else if (j == 1) {
                return _ST2_21;
            } else if (j == 2) {
                return _ST2_22;
            } else {
                assert(false);
            }
        }
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

} // end namespace ttk
