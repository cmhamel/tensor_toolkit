#pragma once
#include <enzyme/enzyme>
#include <gtest/gtest.h>
#include "simple_motions.hpp"
#include <tensor_toolkit.hpp>

namespace ttk_material_model_tests {

template<typename T>
using Duplicated = enzyme::Duplicated<T>;
using SymmetricTensor2 = ttk::SymmetricTensor2<double, ttk::CARTESIAN, 3>;
using Tensor2 = ttk::Tensor2<double, ttk::CARTESIAN, 3>;
using Tensor4 = ttk::Tensor4<double, ttk::CARTESIAN, 3>;

template<typename T>
class SimpleShear {
public:
    TTK_INLINE
    constexpr SimpleShear() = default;
    ~SimpleShear() = default;

    TTK_INLINE
    Tensor2 deformationGradient(double shear) const {
        Tensor2 F({
            1, shear, 0,
            0, 1,     0,
            0, 0,     1
        });
        return F;
    }

    TTK_INLINE
    Tensor2 displacementGradient(double shear) const {
        Tensor2 I(ttk::identity<Tensor2>());
        return deformationGradient(shear) - I;
    }
};

template<typename T>
class UniaxialStrain {
public:
    TTK_INLINE
    constexpr UniaxialStrain() = default;
    ~UniaxialStrain() = default;

    TTK_INLINE
    Tensor2 deformationGradient(double stretch) const {
        Tensor2 F({
            stretch, 0, 0,
            0,       1, 0,
            0,       0, 1
        });
        return F;
    }

    TTK_INLINE
    Tensor2 displacementGradient(double stretch) const {
        Tensor2 I(ttk::identity<Tensor2>());
        return deformationGradient(stretch) - I;
    }
};

class Material {
public:
    TTK_INLINE
    Material() = default;

    TTK_INLINE
    constexpr Material(int numProps_, int numStateVars_)
        : numProps(numProps_),
          numStateVars(numStateVars_) {}

    ~Material() = default;

    TTK_INLINE
    virtual void energy(double& psi, Tensor2& gradU) const = 0;
private:
    int numProps;
    int numStateVars;
};

class NeoHookean : public Material {
public:
    TTK_INLINE
    NeoHookean() = default;

    TTK_INLINE
    constexpr NeoHookean(std::vector<double>& props_)
        : Material(props_.size(), 0),
          props(props_.data()) {
    }

    TTK_INLINE
    void energy(double& psi, Tensor2& gradU) const final {
        Tensor2 I(ttk::identity<Tensor2>());
        Tensor2 F = gradU + I;
        double K = props[0];
        double G = props[1];
        double J = ttk::det(F);
        double J_minus_13 = ttk::cbrt(1.0 / J);
        double J_minus_23 = J_minus_13 * J_minus_13;
        SymmetricTensor2 B = ttk::dott(F);
        double I1_bar = J_minus_23 * ttk::trace(B);
        psi = 0.5 * K * (0.5 * (J * J - 1.0) - ttk::log(J)) +
              0.5 * G * (I1_bar - 3.0);
    }

    double* props;
};

template<typename Mat>
void energy(Mat* mat, double& psi, Tensor2& gradU) {
    mat->energy(psi, gradU);
}

template<typename Mat>
void pk1_stress(
    Mat* mat,
    double& psi,
    double& dpsi,
    Tensor2& gradU,
    Tensor2& P
) {
    psi = 0.0;
    dpsi = 1.0;
    __enzyme_autodiff(
        (void*) energy<Mat>,
        enzyme_const, mat,
        enzyme_dup, &psi, &dpsi,
        enzyme_dup, &gradU, &P
    );
}

template<typename Mat>
void material_tangent(
    Mat* mat,
    Tensor2& gradU,
    double& psi,
    Tensor2& P,
    Tensor4& A
) {
    psi = 0.0;
    double bpsi = 1.0;
    double dpsi = 0.0;
    double dbpsi = 0.0;

    Tensor2 dgradU;
    Tensor2 dP;
 
    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            dgradU.fill(0.0);
            dgradU(k, l) = 1.0;
            P.fill(0.0);
            dP.fill(0.0);
            __enzyme_fwddiff<void>(
                (void*) pk1_stress<Mat>,
                enzyme_const, mat,
                enzyme_dup, &psi, &bpsi,
                enzyme_dup, &dpsi, &dbpsi,
                enzyme_dup, &gradU, &dgradU,
                enzyme_dup, &P, &dP
            );
            std::cout << "gradU   = " << gradU << std::endl;
            std::cout << "dgradU  = " << dgradU << std::endl;
            std::cout << "P       = " << P << std::endl;
            std::cout << "dP      = " << dP << std::endl;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    A(i, j, k, l) = dP(i, j);
                }
            }
        }
    }
}

TEST(TTKMaterialUnitTests, NeoHookeanNoUniaxialStrainByRef) {
    UniaxialStrain<double>* motion = 
        new UniaxialStrain<double>();
    std::vector<double> props = {100.0, 1.0};
    std::vector<double> dprops = {0.0, 0.0};
    NeoHookean* mat = new NeoHookean(props);
    NeoHookean* dmat = new NeoHookean(dprops);
    double psi = 0.0;
    double dpsi = 1.0;
    Tensor2 gradU = motion->displacementGradient(1.5);
    Tensor2 P;
    // P.fill(0.0);
    Tensor4 A;
    // A.fill(0.0);
    gradU = motion->displacementGradient(1.5);
    // energy(mat, psi, gradU);
    // psi = energy(mat, gradU);
    energy(mat, psi, gradU);
    std::cout << "Psi = " << psi << std::endl;
    // pk1_stress(mat, psi, dpsi, gradU, P);
    pk1_stress(mat, psi, dpsi, gradU, P);
    std::cout << "Psi = " << psi << std::endl;
    std::cout << "P   = " << std::endl << P << std::endl;
    material_tangent(mat, gradU, psi, P, A);
    // std::cout << "Psi = " << psi << std::endl;
    // std::cout << "P   = " << std::endl << P << std::endl;
    std::cout << "A   = " << std::endl << A << std::endl;

    // std::cout << "mat.K = " << mat->props[0] << std::endl;
    // std::cout << "mat.G = " << mat->props[1] << std::endl;
    // std::cout << "dmat.K = " << dmat->props[0] << std::endl;
    // std::cout << "dmat.G = " << dmat->props[1] << std::endl;
}

} // end namspace
