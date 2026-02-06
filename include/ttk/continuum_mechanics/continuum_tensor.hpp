#pragma once

#include <string_view>
#include <ttk/core/globals.hpp>
#include <ttk/core/indexing.hpp>
#include <ttk/core/lengths.hpp>
#include <ttk/core/macros.hpp>
#include <ttk/core/tensor.hpp>

namespace ttk {

// specialization for continuum tensors
template<typename T, int M, int D, int O, bool... Args>
struct ContinuumTensor {
public:
    static constexpr int Dimension = D;
    static constexpr int Length = TensorLength<D, O, Args...>::value;
    static constexpr int MetricType = M;
    static constexpr int Order = O;
    static constexpr size_t NumSymArgs             = sizeof...(Args);
    static constexpr bool SymArgs[sizeof...(Args)] = { Args... };
    using TensorType = Tensor<T, M, D, O, Args...>;
    using ValueType = T;

    // using Base = ContinuumTensor<T, M, D, O, Args...>;
    // using Base::operator=;

    TTK_DEFAULTED_FUNCTION
    constexpr ContinuumTensor() = default;

    TTK_FUNCTION
    constexpr ContinuumTensor(const std::initializer_list<T>& data_) {
        assert(data_.size() == Length);
        std::copy(data_.begin(), data_.end(), data.data);
    }

    TTK_FUNCTION
    constexpr ContinuumTensor(const Tensor<T, M, D, O, Args...>& data_)
        : data(data_) {}

    TTK_FUNCTION
    void fill(T value) {
        data.fill(value);
    }

    // below we have two different getData methods
    // one that keeps things fixed and the other
    // that allows for mutability
    const Tensor<T, M, D, O, Args...>& getDataConst() const {
        return data;
    }

    Tensor<T, M, D, O, Args...>& getData() {
        return data;
    }

    TTK_FUNCTION
    constexpr ContinuumTensor<T, M, D, O, Args...>& 
    operator=(const Tensor<T, M, D, O, Args...>& data_) noexcept {
        data = data_;
        return *this;
    }

    template<typename... Ids>
    TTK_FUNCTION
    constexpr T& operator()(Ids... ids) {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data(ids...);
    }

    template<typename... Ids>
    TTK_FUNCTION
    constexpr const T& operator()(Ids... ids) const {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data(ids...);
    }

    // TTK_FUNCTION
    // Tensor operator+(const Tensor& B) const {
    //     Tensor C;
    //     for (int i = 0; i < Length; ++i) {
    //         C.data[i] = this->data[i] + B.data[i];
    //     }
    //     return C;
    // }

    // TTK_FUNCTION
    // ContinuumTensor operator-(const ContinuumTensor& B) const {
    //     ContinuumTensor C;
    //     for (int i = 0; i < Length; ++i) {
    //         C.data[i] = this->data[i] - B.data[i];
    //     }
    //     return C;
    //     // return this->getData() - B.getData();
    // }

    // TTK_FUNCTION
    // Tensor operator*(const T& val) const {
    //     Tensor C;
    //     for (int i = 0; i < Length; ++i) {
    //         C.data[i] = val * this->data[i];
    //     }
    //     return C;
    // }
protected:
    Tensor<T, M, D, O, Args...> data;
};

template<typename T, int M, int D, int O, bool... Syms>
std::ostream& operator<<(std::ostream& os, const ContinuumTensor<T, M, D, O, Syms...>& A) {
    os << A.getDataConst();
    return os;
}

// specialization for continuum vectors
template<typename T, int M, int D>
using ContinuumVector = ContinuumTensor<T, M, D, 1>;

template<typename T, int M, int D>
struct MaterialVector : public ContinuumVector<T, M, D> {
    using ContinuumVector<T, M, D>::ContinuumVector;
    using ContinuumVector<T, M, D>::operator=;
};

template<typename T, int M, int D>
struct SpatialVector : public ContinuumVector<T, M, D> {
    using ContinuumVector<T, M, D>::ContinuumVector;
    using ContinuumVector<T, M, D>::operator=;
};

// specialization for continuum second ordertensors
template<typename T, int M, int D, bool Sym>
using ContinuumTensor2 = ContinuumTensor<T, M, D, 2, Sym>;

template<typename T, int M, int D>
struct MaterialTensor2 : public ContinuumTensor2<T, M, D, true> {};

template<typename T, int M, int D>
struct SpatialTensor2 : public ContinuumTensor2<T, M, D, true> {};

template<typename T, int M, int D>
struct TwoPointTensor2 : public ContinuumTensor2<T, M, D, false> {};

// e.g. tranpose of deformation gradient
template<typename T, int M, int D>
struct TwoPointTensor2T : public ContinuumTensor2<T, M, D, false> {};

template<typename T, int M, int D, bool... Syms>
using ContinuumTensor3 = ContinuumTensor<T, M, D, 3, Syms...>;

// specialization for continuum second ordertensors
template<typename T, int M, int D, bool... Syms>
using ContinuumTensor4 = ContinuumTensor<T, M, D, 4, Syms...>;

template<typename T, int M, int D>
struct TwoPointTensor4 : public ContinuumTensor4<T, M, D, false, false, false> {};

} // end namespace ttk
