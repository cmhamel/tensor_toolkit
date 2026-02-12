#pragma once
#include <ttk/continuum_mechanics/continuum_tensor.hpp>

namespace ttk {

template<Scalar T, int M, int D>
struct SimpleMotion {
public:
    TTK_FUNCTION
    virtual Tensor2<T, M, D> deformationGradient(T) const {
        Tensor2<T, M, D> F;
        F.fill(0.0);
        return F;
    };

    TTK_FUNCTION
    Tensor2<T, M, D> displacementGradient(T val) const {
       Tensor2<T, M, D> I(ttk::identity<Tensor2<T, M, D>>());
       return deformationGradient(val) - I; 
    }
};

template<Scalar T, int M, int D>
struct SimpleShear : public SimpleMotion<T, M, D> {
public:
    TTK_FUNCTION
    Tensor2<T, M, D> deformationGradient(T shear) const final {
        Tensor2<T, M, D> F({
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
    Tensor2<T, M, D> deformationGradient(T stretch) const final {
        Tensor2<T, M, D> F({
            stretch, 0, 0,
            0,       1, 0,
            0,       0, 1
        });
        return F;
    }
};

} // end namespace ttk
