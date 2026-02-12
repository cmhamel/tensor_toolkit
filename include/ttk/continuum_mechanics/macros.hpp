#pragma once
#include <ttk/continuum_mechanics/traits.hpp>
#include <ttk/core/macros.hpp>

namespace ttk {

// compile time error checking macros
#define TTK_OBJECTIVITY_ERROR_MSG(FUNC, ...) \
    static_assert(\
        ttk::__always_false<__VA_ARGS__>,\
        "\n\n\n======================= OBJECTIVITY ERROR =======================\n\n\n"\
        "File    : " __FILE__ "\n"\
        "Function: " FUNC "\n\n"\
        "You attempted to perform an operation on two "\
        "tensors in\ndifferent spaces.\n\n" \
        "An example of this is a dot product "\
        "between\nspatial and material tensors.\n\n\n"\
        "Check immediately above this error message for\n"\
        "the troublesome types and stack trace.\n"\
        "\n\n\n=================================================================\n\n\n"\
    )

#define TTK_OBJECTIVITY_GUARD(FUNC, FUNC_STR) \
    template<typename A, typename B> \
    TTK_FUNCTION \
    auto FUNC(const A&, const B&) { \
        TTK_OBJECTIVITY_ERROR_MSG(FUNC_STR, A, B); \
    }

#define TTK_CONTINUUM_TENSOR_BINARY_OP(FUNC, FUNC_STR, IN1, IN2, OUT)\
    template<typename T, int M, int D>\
    TTK_FUNCTION\
    OUT<T, M, D> FUNC(\
        const IN1<T, M, D>& A,\
        const IN2<T, M, D>& B\
    ) requires IsObjective<IN1<T, M, D>, IN2<T, M, D>> {\
        return FUNC(A.getDataConst(), B.getDataConst());\
    }


#define TTK_CONTINUUM_TENSOR_BINARY_OP_SCALAR(FUNC, FUNC_STR, IN1, IN2)\
    template<typename T, int M, int D>\
    TTK_FUNCTION\
    T FUNC(\
        const IN1<T, M, D>& A,\
        const IN2<T, M, D>& B\
    ) requires IsObjective<IN1<T, M, D>, IN2<T, M, D>> {\
        return FUNC(A.getDataConst(), B.getDataConst());\
    }

#define TTK_CONTINUUM_TENSOR_UNARY_OP(FUNC, IN, OUT)\
    template<typename T, int M, int D>\
    TTK_FUNCTION\
    OUT<T, M, D> FUNC(const IN<T, M, D>& A) {\
        return FUNC(A.getDataConst());\
    }

#define TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(FUNC, IN)\
    template<typename T, int M, int D>\
    TTK_FUNCTION\
    T FUNC(const IN<T, M, D>& A) {\
        return FUNC(A.getDataConst());\
    }



} // end namespace ttk
