#pragma once
#include <cassert> // for assert (TODO replace with better error handling)
#include <iomanip>
#include "indexing.hpp"
#include "lengths.hpp"
#include "macros.hpp"

namespace ttk {

struct _EmptyTensor {};

TTK_FORCEINLINE_FUNCTION
constexpr _EmptyTensor EmptyTensor() {
    return {};
}

template<typename T, int M, int D, int O, bool... Args>
class Tensor {
public:
    // raw template arg helps
    static constexpr int Dimension                 = D;
    static constexpr int Length                    = TensorLength<D, O, Args...>::value;
    static constexpr int MetricType                = M;
    static constexpr int Order                     = O;
    static constexpr size_t NumSymArgs             = sizeof...(Args);
    static constexpr bool SymArgs[sizeof...(Args)] = { Args... };
    using ValueType = T;

    // derived template args

    // TTK_DEFAULTED_FUNCTION
    TTK_FORCEINLINE_FUNCTION
    constexpr Tensor() = default;

    TTK_FORCEINLINE_FUNCTION
    constexpr explicit Tensor(const _EmptyTensor&) : data{} {}

    TTK_FUNCTION
    constexpr explicit Tensor(T value) {
        fill(value);
    }

    // TODO fix assert
    // TTK_FUNCTION
    // constexpr Tensor(std::initializer_list<T> data_) {
    //     assert(data_.size() == Length);
    //     std::copy(data_.begin(), data_.end(), data);
    // }
    template <size_t N>
    constexpr Tensor(const T (&arr)[N]) {
        static_assert(N == Length);
        for (size_t i = 0; i < N; ++i) {
            data[i] = arr[i];
        }
    }

    TTK_FUNCTION
    void fill(T value) {
        std::fill(std::begin(data), std::end(data), value);
    }

    TTK_FUNCTION
    const T* getDataConst() const {
        return data;
    }

    template<typename... Ids>
    TTK_FUNCTION
    constexpr int linearIndex(Ids... ids) const {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return LinearIndex<O, Args...>::compute(D, ids...);
    }

    // operator overloading
    // template<typename... Ids>
    // TTK_FUNCTION
    // T& operator()(Ids... ids) const {
    //     static_assert(
    //         sizeof...(Ids) == O,
    //         "Number of indices must equal tensor order"
    //     );
    //     return const_cast<T&>(data[linear_index(ids...)]);
    // }
    // template<typename... Ids>
    // TTK_FUNCTION
    // T operator()(Ids... ids) const {
    //     static_assert(
    //         sizeof...(Ids) == O,
    //         "Number of indices must equal tensor order"
    //     );
    //     return data[linear_index(ids...)];
    // }

    template<typename... Ids>
    TTK_FUNCTION
    T& operator()(Ids... ids) {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data[linearIndex(ids...)];
    }

    template<typename... Ids>
    TTK_FUNCTION
    const T& operator()(Ids... ids) const {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data[linearIndex(ids...)];
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

    TTK_FUNCTION
    Tensor operator/(const T& val) const {
        Tensor C;
        for (int i = 0; i < Length; ++i) {
            C.data[i] = this->data[i] / val;
        }
        return C;
    }

    TTK_FUNCTION
    bool operator==(const Tensor& A) const {
        bool equal = false;
        for (int i = 0; i < Length; ++i) {
            equal = equal && (this->data[i] == A.data[i]);
        }
        return equal;
    }

// private:
    T data[Length];
};

} // end namespace ttk
