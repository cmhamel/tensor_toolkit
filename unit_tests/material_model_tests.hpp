#pragma once

#if TENSOR_TOOLKIT_ENABLE_ENZYME
#include <enzyme/enzyme>
#include <gtest/gtest.h>
#include <ttk/tensor_toolkit.hpp>

namespace ttk_material_model_tests {

template<typename T>
using Duplicated = enzyme::Duplicated<T>;
using SymmetricTensor2 = ttk::SymmetricTensor2<double, ttk::CARTESIAN, 3>;
using Tensor2 = ttk::Tensor2<double, ttk::CARTESIAN, 3>;
using Tensor4 = ttk::Tensor4<double, ttk::CARTESIAN, 3>;

using TwoPointTensor2 = ttk::TwoPointTensor2<double, ttk::CARTESIAN, 3>;

class Material {
public:
    TTK_DEFAULTED_FUNCTION
    Material() = default;

    ~Material() = default;

    TTK_FUNCTION
    virtual void energy(double& psi, TwoPointTensor2& gradU) const = 0;
};

class NeoHookean : public Material {
public:
    TTK_DEFAULTED_FUNCTION
    NeoHookean() = default;

    TTK_FUNCTION
    constexpr NeoHookean(std::vector<double>& props_)
        : Material(),
          props(props_.data()) {
    }

    TTK_FUNCTION
    void energy(double& psi, TwoPointTensor2& gradU) const final {
        TwoPointTensor2 I(ttk::identity<TwoPointTensor2>());
        TwoPointTensor2 F = gradU + I;
        double K = props[0];
        double G = props[1];
        double J = ttk::det(F);
        double J_minus_13 = ttk::cbrt(1.0 / J);
        double J_minus_23 = J_minus_13 * J_minus_13;
        double I1_bar = J_minus_23 * ttk::first_invariant(F);
        psi = 0.5 * K * (0.5 * (J * J - 1.0) - ttk::log(J)) +
              0.5 * G * (I1_bar - 3.0);
    }

    double* props;
};

template<typename Mat>
void energy(Mat* mat, double& psi, TwoPointTensor2& gradU) {
    mat->energy(psi, gradU);
}

template<typename Mat>
void pk1_stress(
    Mat* mat,
    double& psi,
    double& dpsi,
    TwoPointTensor2& gradU,
    TwoPointTensor2& P
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
    TwoPointTensor2& gradU,
    double& psi,
    TwoPointTensor2& P,
    Tensor4& A
) {
    psi = 0.0;
    double bpsi = 1.0;
    double dpsi = 0.0;
    double dbpsi = 0.0;

    TwoPointTensor2 dgradU;
    TwoPointTensor2 dP;
 
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
            // std::cout << "gradU   = " << gradU << std::endl;
            // std::cout << "dgradU  = " << dgradU << std::endl;
            // std::cout << "P       = " << P << std::endl;
            // std::cout << "dP      = " << dP << std::endl;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    A(i, j, k, l) = dP(i, j);
                }
            }
        }
    }
}

TEST(TTKMaterialUnitTests, NeoHookeanNoUniaxialStrainByRef) {
    ttk::UniaxialStrain<double, ttk::CARTESIAN, 3>* motion = 
        new ttk::UniaxialStrain<double, ttk::CARTESIAN, 3>();
    std::vector<double> props = {100.0, 1.0};
    std::vector<double> dprops = {0.0, 0.0};
    NeoHookean* mat = new NeoHookean(props);
    // NeoHookean* dmat = new NeoHookean(dprops);
    double psi = 0.0;
    double dpsi = 1.0;
    TwoPointTensor2 gradU = motion->displacementGradient(1.5);
    TwoPointTensor2 P;
    P.fill(0.0);
    Tensor4 A;
    A.fill(0.0);
    gradU = motion->displacementGradient(1.5);
    energy(mat, psi, gradU);
    pk1_stress(mat, psi, dpsi, gradU, P);
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

#endif
