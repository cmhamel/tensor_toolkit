#pragma once

#include <string_view>
#include <ttk/core/macros.hpp>
#include <ttk/core/tensor.hpp>

namespace ttk {

// specialization for continuum tensors
template<typename Derived, Scalar T, int M, int D, int O, int... Args>
struct ContinuumTensor {
public:
    static constexpr int Dimension                = D;
    static constexpr int Length                   = TensorLength<D, O, Args...>::value;
    static constexpr int MetricType               = M;
    static constexpr int Order                    = O;
    static constexpr size_t NumSymArgs            = sizeof...(Args);
    static constexpr int SymArgs[sizeof...(Args)] = { Args... };
    using TensorType = Tensor<T, M, D, O, Args...>;
    using ValueType = T;

    // using Base = ContinuumTensor<T, M, D, O, Args...>;
    // using Base::operator=;

    TTK_FORCEINLINE_FUNCTION
    constexpr ContinuumTensor() = default;

    TTK_FORCEINLINE_FUNCTION
    constexpr ContinuumTensor(T value) {
        fill(value);
    }

    TTK_FORCEINLINE_FUNCTION
    constexpr ContinuumTensor(const std::initializer_list<T>& data_) {
        assert(data_.size() == Length);
        std::copy(data_.begin(), data_.end(), data.data);
    }

    TTK_FUNCTION
    constexpr ContinuumTensor(const TensorType& data_)
        : data(data_) {}

    TTK_FUNCTION
    void fill(T value) {
        data.fill(value);
    }

    TTK_FORCEINLINE_FUNCTION
    constexpr Derived& derived() {
        return static_cast<Derived&>(*this);
    }

    TTK_FORCEINLINE_FUNCTION
    constexpr const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

    // below we have two different getData methods
    // one that keeps things fixed and the other
    // that allows for mutability
    const TensorType& getDataConst() const {
        return data;
    }

    TensorType& getData() {
        return data;
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

    TTK_FUNCTION
    constexpr ContinuumTensor<Derived, T, M, D, O, Args...>& 
    operator=(const TensorType& data_) noexcept {
        data = data_;
        return *this;
    }

    TTK_FUNCTION
    constexpr Derived operator+(const Derived& B) const {
        return Derived(data + B.getDataConst());
    }

    TTK_FUNCTION
    constexpr Derived operator-(const Derived& B) const {
        return Derived(data - B.getDataConst());
    }

protected:
    TensorType data;
};

template<typename Derived, Scalar T, int M, int D, int O, int... Syms>
std::ostream& operator<<(std::ostream& os, const ContinuumTensor<Derived, T, M, D, O, Syms...>& A) {
    os << A.getDataConst();
    return os;
}

// specialization for continuum vectors
template<typename Derived, Scalar T, int M, int D>
using ContinuumVector = ContinuumTensor<Derived, T, M, D, 1>;

template<Scalar T, int M, int D>
struct MaterialVector : public ContinuumVector<MaterialVector<T, M, D>, T, M, D> {
    using ContinuumVector<MaterialVector<T, M, D>, T, M, D>::ContinuumVector;
    // using ContinuumVector<T, M, D>::operator=;
};

template<Scalar T, int M, int D>
struct SpatialVector : public ContinuumVector<SpatialVector<T, M, D>, T, M, D> {
    using ContinuumVector<SpatialVector<T, M, D>, T, M, D>::ContinuumVector;
    // using ContinuumVector<T, M, D>::operator=;
};

// specialization for continuum second ordertensors
template<typename Derived, Scalar T, int M, int D, int Sym>
using ContinuumTensor2 = ContinuumTensor<Derived, T, M, D, 2, Sym>;

template<Scalar T, int M, int D>
struct MaterialTensor2 : public ContinuumTensor2<MaterialTensor2<T, M, D>, T, M, D, SYMM> {
    using ContinuumTensor2<MaterialTensor2<T, M, D>, T, M, D, SYMM>::ContinuumTensor2;
};

template<Scalar T, int M, int D>
struct SpatialTensor2 : public ContinuumTensor2<SpatialTensor2<T, M, D>, T, M, D, SYMM> {
    using ContinuumTensor2<SpatialTensor2<T, M, D>, T, M, D, SYMM>::ContinuumTensor2;
};

template<Scalar T, int M, int D>
struct TwoPointTensor2 : public ContinuumTensor2<TwoPointTensor2<T, M, D>, T, M, D, FULL> {
    using ContinuumTensor2<TwoPointTensor2<T, M, D>, T, M, D, FULL>::ContinuumTensor2;
};

// e.g. tranpose of deformation gradient
template<Scalar T, int M, int D>
struct TwoPointTensor2T : public ContinuumTensor2<TwoPointTensor2T<T, M, D>, T, M, D, FULL> {
    using ContinuumTensor2<TwoPointTensor2T<T, M, D>, T, M, D, FULL>::ContinuumTensor2;
};

template<typename Derived, Scalar T, int M, int D, int... Syms>
using ContinuumTensor3 = ContinuumTensor<Derived, T, M, D, 3, Syms...>;

// specialization for continuum second ordertensors
template<typename Derived, Scalar T, int M, int D, int... Syms>
using ContinuumTensor4 = ContinuumTensor<Derived, T, M, D, 4, Syms...>;

template<Scalar T, int M, int D>
struct TwoPointTensor4 : public ContinuumTensor4<TwoPointTensor4<T, M, D>, T, M, D, FULL, FULL, FULL> {};

} // end namespace ttk
