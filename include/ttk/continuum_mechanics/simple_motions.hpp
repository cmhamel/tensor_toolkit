#pragma once
#include <ttk/continuum_mechanics/continuum_tensor.hpp>
#include <ttk/core/unary_operators.hpp>

namespace ttk {

template<Scalar T, int M, int D>
struct SimpleMotion {
public:
    TTK_FUNCTION
    virtual TwoPointTensor2<T, M, D> deformationGradient(T) const {
        TwoPointTensor2<T, M, D> F(ttk::identity<TwoPointTensor2<T, M, D>>());
        return F;
    };

    virtual TwoPointTensor2<T, M, D> deformationGradientRate(T) const {
        TwoPointTensor2<T, M, D> Fdot(0.0);
        return Fdot;
    }

    TTK_FUNCTION
    TwoPointTensor2<T, M, D> displacementGradient(T val) const {
       TwoPointTensor2<T, M, D> I(ttk::identity<TwoPointTensor2<T, M, D>>());
       return deformationGradient(val) - I; 
    }

    // TTK_FUNCTION
    // SpatialTensor2<T, M, D> velocityGradient(T val, T valDot) const {
    //     TwoPointTensor2<T, M, D> F = deformationGradient(val);
    //     TwoPointTensor2<T, M, D> Fdot = deformationGradientRate(valDot);
    //     TwoPointTensor2<T, M, D> Fdot_inv = ttk::inv(Fdot);
    //     return ttk::dot(F, Fdot_inv);
    // }

    // TTK_FUNCTION
    // SymmetricTensor2<T, M, D> symmetricVelocityGradient(T val, T valDot) const {
    //     return ttk::symmetric(velocityGradient(val, valDot));
    // }
};

template<Scalar T, int M, int D>
struct SimpleShear : public SimpleMotion<T, M, D> {
public:
    TTK_FUNCTION
    TwoPointTensor2<T, M, D> deformationGradient(T shear) const final {
        TwoPointTensor2<T, M, D> F({
            1, shear, 0,
            0, 1,     0,
            0, 0,     1
        });
        return F;
    }
};

template<Scalar T, int M, int D>
struct UniaxialStrain : public SimpleMotion<T, M, D> {
public:
    TTK_FUNCTION
    TwoPointTensor2<T, M, D> deformationGradient(T stretch) const final {
        TwoPointTensor2<T, M, D> F({
            stretch, 0, 0,
            0,       1, 0,
            0,       0, 1
        });
        return F;
    }
};

} // end namespace ttk
