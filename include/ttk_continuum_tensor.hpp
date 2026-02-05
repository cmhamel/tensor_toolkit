#pragma once
#include <core/ttk_globals.hpp>
#include <core/ttk_indexing.hpp>
#include <core/ttk_lengths.hpp>
#include <core/ttk_macros.hpp>
#include <core/ttk_tensor.hpp>
#include <string_view>

namespace ttk {


template<typename A, typename B>
concept SameFrame = std::is_same_v<A, B>;

// std::string __objectivity_error(
//     std::string& funcName, bool isBinaryOP
// ) {
//     // std::ostream err;
//     std::string err = "\n\n\nObjectivity Error!!!\n\n\n";
//     // err << "You attempted to call " << funcName << " on "; 
//     return err;
// }
// constexpr std::string_view ttk::type_name() [T = MaterialVector<double,0,3>]


// specialization for continuum tensors
template<typename T, int M, int D, int O, bool... Args>
struct ContinuumTensor {
public:
    static constexpr int Dimension = D;
    static constexpr int Length = TensorLength<D, O, Args...>::value;
    static constexpr int MetricType = M;
    static constexpr int Order = O;
    static constexpr bool SymArgs[sizeof...(Args)] = { Args... };
    using TensorType = Tensor<T, M, D, O, Args...>;
    using ValueType = T;

    // using Base = ContinuumTensor<T, M, D, O, Args...>;
    // using Base::operator=;

    TTK_DEFAULTED_FUNCTION
    ContinuumTensor() = default;

    TTK_FUNCTION
    ContinuumTensor(const std::initializer_list<T>& data_) {
        assert(data_.size() == Length);
        std::copy(data_.begin(), data_.end(), data.getData());
    }

    TTK_FUNCTION
    ContinuumTensor(const Tensor<T, M, D, O, Args...>& data_)
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
    static constexpr int getLength() {
        return Length;
    }

    TTK_FUNCTION
    constexpr ContinuumTensor<T, M, D, O, Args...>& 
    operator=(const Tensor<T, M, D, O, Args...>& data_) noexcept {
        data = data_;
        return *this;
    }

    template<typename... Ids>
    TTK_FUNCTION
    T operator()(Ids... ids) const {
        static_assert(
            sizeof...(Ids) == O,
            "Number of indices must equal tensor order"
        );
        return data(ids...);
    }

    template<typename... Ids>
    TTK_FUNCTION
    T& operator()(Ids... ids) {
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

// specialization for continuum second ordertensors
template<typename T, int M, int D, bool... Syms>
using ContinuumTensor4 = ContinuumTensor<T, M, D, 4, Syms...>;

template<typename T, int M, int D>
struct TwoPointTensor4 : public ContinuumTensor4<T, M, D, false, false, false> {};


// #define TTK_OBJECTIVITY_ERROR(FUNC, TYPE1, TYPE2, MESSAGE)          \
//     static_assert(                                                      \
//         __always_false<TYPE1, TYPE2>, \
//         "\n\n\nObjectivity error!\n\n\n" \
//         "You attempted to call " FUNC \
//         " on incompatible tensor types:\n" \
//         "\t" std::string(::ttk::__tensor_type_name<TYPE1>()) " and\n" \
//         "\t" std::string(::ttk::__tensor_type_name<TYPE2>()) ".\n\n" \
//         MESSAGE \
//     )
// #define TTK_OBJECTIVITY_ERROR(FUNC, TYPE1, TYPE2, MESSAGE)                  \
//     static_assert(                                                          \
//         __always_false<TYPE1, TYPE2>,                                       \
//         "Objectivity error! You attempted to call " FUNC                    \
//         " on incompatible tensor types:\n  A = "                             \
//         __tensor_type_name<TYPE1>()                                         \
//         "\n  B = "                                                          \
//         __tensor_type_name<TYPE2>()                                         \
//         "\n" MESSAGE                                                        \
//     )


template<typename T>
struct __BaseTensorType {
    using type = T;
};

// specialization for ContinuumTensor wrappers
template<typename T, int M, int D, int O, bool... Args>
struct __BaseTensorType<ContinuumTensor<T, M, D, O, Args...>> {
    using type = ContinuumTensor<T, M, D, O, Args...>;
};

// helper alias
template<typename T>
using __BaseTensorTypeT = typename __BaseTensorType<T>::type;

template<typename T>
constexpr const char* __tensor_type_name() {
    using Base = __BaseTensorTypeT<T>;
    // return __PRETTY_FUNCTION__;
// }
#if defined(__clang__) || defined(__GNUC__)
    return __PRETTY_FUNCTION__; // will include Base
#elif defined(_MSC_VER)
    return __FUNCSIG__;
#else
    return "type_name not supported";
#endif
}

// template<typename T1, typename T2>
// constexpr const char* __objectivity_error_msg(const char* func, const char* op_type) {
//     // __PRETTY_FUNCTION__ already includes T1, T2, so just return a nicely formatted message
// #if defined(__clang__) || defined(__GNUC__)
//     return 
//         "\n==================== OBJECTIVITY ERROR ====================\n"
//         "Function: " 
//         // NOTE: T1 and T2 types printed in __PRETTY_FUNCTION__
//         // " " + func + "\n"
//         "Tensor types involved:\n"
//         "  A = " + __tensor_type_name<T1>() + "\n"
//         "  B = " + __tensor_type_name<T2>() + "\n"
//         "===========================================================\n";
// #else
//     return "Objectivity error (type info unavailable on this compiler)";
// #endif
// }


#define TTK_OBJECTIVITY_ERROR(FUNC, TYPE1, TYPE2)\
    static_assert(\
        ttk::__always_false<TYPE1, TYPE2>,\
        "\n\n\n==================== OBJECTIVITY ERROR ====================\n\n\n"\
        "File    : " __FILE__ "\n"\
        "Function: " FUNC "\n"\
        "\n\n\n===========================================================\n\n\n"\
    )

} // end namespace ttk
