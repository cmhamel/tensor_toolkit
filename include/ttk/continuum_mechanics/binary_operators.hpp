#pragma once
#include <ttk/core/binary_operators.hpp>
#include <ttk/continuum_mechanics/continuum_tensor.hpp>
#include <ttk/continuum_mechanics/macros.hpp>

namespace ttk {

// cross product
TTK_OBJECTIVITY_GUARD(cross, "cross")
TTK_CONTINUUM_TENSOR_BINARY_OP(cross, "cross", MaterialVector, MaterialVector, MaterialVector);
TTK_CONTINUUM_TENSOR_BINARY_OP(cross, "cross", SpatialVector, SpatialVector, SpatialVector);
// dot
TTK_OBJECTIVITY_GUARD(dot, "dot");
TTK_CONTINUUM_TENSOR_BINARY_OP_SCALAR(dot, "dot", MaterialVector, MaterialVector);
TTK_CONTINUUM_TENSOR_BINARY_OP_SCALAR(dot, "dot", SpatialVector, SpatialVector);
// alot more to do here
// outer products
TTK_OBJECTIVITY_GUARD(otimes, "otimes")

// otimesl
TTK_OBJECTIVITY_GUARD(otimesl, "otimesl")

// otimesu
TTK_OBJECTIVITY_GUARD(otimesu, "otimesu")

// basic operations
// template<Scalar T, int M, int D> 
// TTK_FUNCTION
// TwoPointTensor2<T, M, D> operator+(
//     const TwoPointTensor2<T, M, D>& A,
//     const TwoPointTensor2<T, M, D>& B
// ) {
//     return TwoPointTensor2<T, M, D>(A.getDataConst() + B.getDataConst());
// }

} // end namespace ttk
