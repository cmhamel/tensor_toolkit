#pragma once
#include <core/ttk_indexing.hpp>
#include <core/ttk_lengths.hpp>
#include <core/ttk_macros.hpp>
#include <core/ttk_tensor.hpp>

namespace ttk {

// specialization for continuum tensors
template<typename T, int M, int D, int O, bool... Args>
struct ContinuumTensor {
public: 
    static constexpr int Length = TensorLength<D, O, Args...>::value;
    using TensorType = Tensor<T, M, D, O, Args...>;

    TTK_INLINE 
    constexpr ContinuumTensor() = default;

    // TTK_INLINE
    // constexpr ContinuumTensor(std::array<T, Length>& data_)
    //     : data(Tensor<T, M, D, O, Args...>(data_)) {}

    TTK_INLINE
    constexpr ContinuumTensor(std::initializer_list<T>& data_) {
        assert(data_.size() == Length);
        std::copy(data_.begin(), data_.end(), data_.begin());
    }

    TTK_INLINE
    constexpr ContinuumTensor(Tensor<T, M, D, O, Args...>& data_)
        : data(data_) {}

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

    TTK_INLINE
    static constexpr int getLength() {
        return Length;
    }

    template<typename... Ids>
    TTK_INLINE
    T operator()(Ids... ids) const {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data(ids...);
    }

    template<typename... Ids>
    TTK_INLINE
    T& operator()(Ids... ids) {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data(ids...);
    }

    // TTK_INLINE
    // Tensor operator+(const Tensor& B) const {
    //     Tensor C;
    //     for (int i = 0; i < Length; ++i) {
    //         C.data[i] = this->data[i] + B.data[i];
    //     }
    //     return C;
    // }

    // TTK_INLINE
    // ContinuumTensor operator-(const ContinuumTensor& B) const {
    //     ContinuumTensor C;
    //     for (int i = 0; i < Length; ++i) {
    //         C.data[i] = this->data[i] - B.data[i];
    //     }
    //     return C;
    //     // return this->getData() - B.getData();
    // }

    // TTK_INLINE
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
struct MaterialVector : public ContinuumVector<T, M, D> {};

template<typename T, int M, int D>
struct SpatialVector : public ContinuumVector<T, M, D> {};

// specialization for continuum second ordertensors
template<typename T, int M, int D, bool Sym>
using ContinuumTensor2 = ContinuumTensor<T, M, D, 2, Sym>;

template<typename T, int M, int D>
struct MaterialTensor2 : public ContinuumTensor2<T, M, D, true> {
    TTK_INLINE
    constexpr MaterialTensor2() = default;

    TTK_INLINE 
    constexpr MaterialTensor2(SymmetricTensor2<T, M, D> data_)
        : ContinuumTensor2<T, M, D, true>(data_) {}
};

template<typename T, int M, int D>
struct SpatialTensor2 : public ContinuumTensor2<T, M, D, true> {
public:
    TTK_INLINE
    constexpr SpatialTensor2() = default;

    TTK_INLINE
    constexpr SpatialTensor2(SymmetricTensor2<T, M, D> data_)
        : ContinuumTensor2<T, M, D, true>(data_) {}
};

template<typename T, int M, int D>
struct TwoPointTensor2 : public ContinuumTensor2<T, M, D, false> {
public:
    TTK_INLINE
    constexpr TwoPointTensor2() = default;

    TTK_INLINE
    constexpr TwoPointTensor2(std::initializer_list<T>& data_)
        : ContinuumTensor2<T, M, D, false>(data_) {}

    TTK_INLINE
    constexpr TwoPointTensor2(Tensor2<T, M, D> data_)
        : ContinuumTensor2<T, M, D, false>(data_) {}

    // TODO improve operators by making them less "simple"
    TTK_INLINE
    TwoPointTensor2 operator-(const TwoPointTensor2& B) const {
        TwoPointTensor2 C;
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                C(i, j) = this->data(i, j) - B(i, j);
            }
        }
        return C;
    }

    TTK_INLINE
    TwoPointTensor2 operator+(const TwoPointTensor2& B) const {
        TwoPointTensor2 C;
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                C(i, j) = this->data(i, j) + B(i, j);
            }
        }
        return C;
    }
};

// specialization for continuum second ordertensors
template<typename T, int M, int D, bool... Syms>
using ContinuumTensor4 = ContinuumTensor<T, M, D, 4, Syms...>;

template<typename T, int M, int D>
struct TwoPointTensor4 : public ContinuumTensor4<T, M, D, false, false, false> {
public:
    TTK_INLINE
    constexpr TwoPointTensor4() = default;

    TTK_INLINE
    constexpr TwoPointTensor4(std::initializer_list<T>& data_)
        : ContinuumTensor4<T, M, D, false, false, false>(data_) {}
    
    TTK_INLINE
    constexpr TwoPointTensor4(Tensor4<T, M, D> data_)
        : ContinuumTensor4<T, M, D, false, false, false>(data_) {}
};

} // end namespace ttk
