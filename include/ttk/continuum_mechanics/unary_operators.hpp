#pragma once
#include <ttk/continuum_mechanics/continuum_tensor.hpp>
#include <ttk/continuum_mechanics/macros.hpp>

namespace ttk {
// det
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(det, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(det, SpatialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(det, TwoPointTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(det, TwoPointTensor2T);
// dev
TTK_CONTINUUM_TENSOR_UNARY_OP(dev, MaterialTensor2, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(dev, SpatialTensor2, SpatialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(dev, TwoPointTensor2, TwoPointTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(dev, TwoPointTensor2T, TwoPointTensor2T);
// dott
TTK_CONTINUUM_TENSOR_UNARY_OP(dott, TwoPointTensor2, SpatialTensor2);
// eigen (defined without macro since its a special case for return type)
// template<Scalar T, int M, int D> 
// TTK_FUNCTION
// auto eigen(const MaterialTensor2<T, M, D>& A) {
//     auto [_evals, _evecs] = eigen(A.getDataConst());
//     MaterialVector<T, M, D> evals()
// }
// template<Scalar T, int M, int D> 
// TTK_FUNCTION
// auto eigen(const SpatialTensor2<T, M, D>& A) {
//     return eigen(A.getDataConst());
// }
// exp
TTK_CONTINUUM_TENSOR_UNARY_OP(exp, MaterialTensor2, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(exp, SpatialTensor2, SpatialTensor2);
// first_invariant
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(first_invariant, TwoPointTensor2);
// log
TTK_CONTINUUM_TENSOR_UNARY_OP(log, MaterialTensor2, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(log, SpatialTensor2, SpatialTensor2);
// norm
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, MaterialVector);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, SpatialVector);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, SpatialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, TwoPointTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, TwoPointTensor2T);
// pow
TTK_CONTINUUM_TENSOR_UNARY_OP(pow, MaterialTensor2, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(pow, SpatialTensor2, SpatialTensor2);
// sqrt
TTK_CONTINUUM_TENSOR_UNARY_OP(sqrt, MaterialTensor2, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(sqrt, SpatialTensor2, SpatialTensor2);
// tdot
TTK_CONTINUUM_TENSOR_UNARY_OP(tdot, TwoPointTensor2, MaterialTensor2);
// trace
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, SpatialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, TwoPointTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, TwoPointTensor2T);
// vol
TTK_CONTINUUM_TENSOR_UNARY_OP(vol, MaterialTensor2, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(vol, SpatialTensor2, SpatialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(vol, TwoPointTensor2, TwoPointTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP(vol, TwoPointTensor2T, TwoPointTensor2T);
} // end namespace ttk
