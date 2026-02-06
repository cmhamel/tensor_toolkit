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
// norm
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, MaterialVector);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, SpatialVector);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, SpatialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, TwoPointTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(norm, TwoPointTensor2T);
// tdot
TTK_CONTINUUM_TENSOR_UNARY_OP(tdot, TwoPointTensor2, MaterialTensor2);
// trace
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, MaterialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, SpatialTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, TwoPointTensor2);
TTK_CONTINUUM_TENSOR_UNARY_OP_SCALAR(trace, TwoPointTensor2T);

} // end namespace ttk
