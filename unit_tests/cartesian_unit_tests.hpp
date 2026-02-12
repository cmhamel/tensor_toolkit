#pragma once
#include <gtest/gtest.h>
#include <ttk/tensor_toolkit.hpp>

namespace ttk_cartesian_tests {

template<typename T, int D>
using SymmetricTensor2 = ttk::SymmetricTensor2<double, ttk::CARTESIAN, D>;
template<typename T, int D>
using Tensor2 = ttk::Tensor2<double, ttk::CARTESIAN, D>;
template<typename T, int D>
using Tensor3 = ttk::Tensor3<double, ttk::CARTESIAN, D>;
template<typename T, int D>
using Tensor4 = ttk::Tensor4<double, ttk::CARTESIAN, D>;
template<typename T, int D>
using Vector = ttk::Vector<double, ttk::CARTESIAN, D>;

Tensor2<double, 3> uniform_with_bounds(double min, double max) {
    Tensor2<double, 3> A(ttk::uniform<Tensor2<double, 3>>());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (A(i, j) < min) A(i, j) = min;
            if (A(i, j) > max) A(i, j) = max;
        }
    }
    return A;
}

template<typename TensorType>
void check_equal(const TensorType& A, const TensorType& B, const double& tol = 1e-10) {
    auto A_data = A.getDataConst();
    auto B_data = B.getDataConst();
    for (int i = 0; i < A.Length; ++i) {
        ASSERT_NEAR(A_data[i], B_data[i], tol);
    }
}

TEST(TTKUnitTests, TestLength2D) {
    Vector<double, 2> vec;
    EXPECT_EQ(vec.Length, 2);

    Tensor2<double, 2> tens2;
    EXPECT_EQ(tens2.Length, 4);

    SymmetricTensor2<double, 2> tens2s;
    EXPECT_EQ(tens2s.Length, 3);

    Tensor3<double, 2> tens3;
    EXPECT_EQ(tens3.Length, 8);

    Tensor4<double, 2> tens4;
    EXPECT_EQ(tens4.Length, 16);
}

TEST(TTKUnitTests, TestLength3D) {
    Vector<double, 3> vec;
    EXPECT_EQ(vec.Length, 3);

    Tensor2<double, 3> tens2;
    EXPECT_EQ(tens2.Length, 9);

    SymmetricTensor2<double, 3> tens2s;
    EXPECT_EQ(tens2s.Length, 6);

    Tensor3<double, 3> tens3;
    EXPECT_EQ(tens3.Length, 27);

    Tensor4<double, 3> tens4;
    EXPECT_EQ(tens4.Length, 81);
}

TEST(TTKUnitTests, TestIndex3D) {
    ttk::Vector<double, ttk::CARTESIAN, 3> vec({{1, 2, 3}});
    EXPECT_EQ(vec(0), 1.0);
    EXPECT_EQ(vec(1), 2.0);
    EXPECT_EQ(vec(2), 3.0);

    ttk::Tensor2<double, ttk::CARTESIAN, 3> tens({1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_EQ(tens(0, 0), 1.0);
    EXPECT_EQ(tens(1, 2), 6.0);
}

TEST(TTKUnitTests, TestVectorInput) {
    Vector<double, 3> v({1, 2, 3});
    EXPECT_NEAR(v(0), 1, 1e-12);
    EXPECT_NEAR(v(1), 2, 1e-12);
    EXPECT_NEAR(v(2), 3, 1e-12);
    std::cout << v << std::endl;
}

TEST(TTKUnitTests, TestTensor2Input) {
    Tensor2<double, 3> A({1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_NEAR(A(0, 0), 1, 1e-12);
    EXPECT_NEAR(A(0, 1), 2, 1e-12);
    EXPECT_NEAR(A(0, 2), 3, 1e-12);

    EXPECT_NEAR(A(1, 0), 4, 1e-12);
    EXPECT_NEAR(A(1, 1), 5, 1e-12);
    EXPECT_NEAR(A(1, 2), 6, 1e-12);

    EXPECT_NEAR(A(2, 0), 7, 1e-12);
    EXPECT_NEAR(A(2, 1), 8, 1e-12);
    EXPECT_NEAR(A(2, 2), 9, 1e-12);
    std::cout << A << std::endl;
}

TEST(TTKUnitTests, TestSymmetricTensor2Input) {
    SymmetricTensor2<double, 3> A({1, 2, 3, 4, 5, 6});
    EXPECT_NEAR(A(0, 0), 1, 1e-12);
    EXPECT_NEAR(A(0, 1), 2, 1e-12);
    EXPECT_NEAR(A(0, 2), 3, 1e-12);

    EXPECT_NEAR(A(1, 0), 2, 1e-12);
    EXPECT_NEAR(A(1, 1), 4, 1e-12);
    EXPECT_NEAR(A(1, 2), 5, 1e-12);

    EXPECT_NEAR(A(2, 0), 3, 1e-12);
    EXPECT_NEAR(A(2, 1), 5, 1e-12);
    EXPECT_NEAR(A(2, 2), 6, 1e-12);
    std::cout << A << std::endl;
}

TEST(TTKUnitTests, TestTensor2Identity) {
    Tensor2<double, 3> I = ttk::identity<Tensor2<double, 3>>();
    EXPECT_NEAR(I(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(I(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(I(2, 2), 1.0, 1e-12);

    EXPECT_NEAR(I(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(I(1, 2), 0.0, 1e-12);
    EXPECT_NEAR(I(2, 0), 0.0, 1e-12);

    EXPECT_NEAR(I(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(I(2, 1), 0.0, 1e-12);
    EXPECT_NEAR(I(0, 2), 0.0, 1e-12);
}

TEST(TTKUnitTests, TestSymmetricTensor2Identity) {
    SymmetricTensor2<double, 3> I = ttk::identity<SymmetricTensor2<double, 3>>();
    EXPECT_NEAR(I(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(I(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(I(2, 2), 1.0, 1e-12);

    EXPECT_NEAR(I(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(I(1, 2), 0.0, 1e-12);
    EXPECT_NEAR(I(2, 0), 0.0, 1e-12);

    EXPECT_NEAR(I(1, 0), 0.0, 1e-12);
    EXPECT_NEAR(I(2, 1), 0.0, 1e-12);
    EXPECT_NEAR(I(0, 2), 0.0, 1e-12);
}

TEST(TTKUnitTests, TestTensor3Input) {
    Tensor3<double, 3> A({
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27
    });
    std::cout << "A length" << A.Length << std::endl;
    std::cout << A << std::endl;
}

TEST(TTKUnitTests, TestTensor4Input) {
    Tensor4<double, 3> A({
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81
    });
    std::cout << "A length" << A.Length << std::endl;
    std::cout << A << std::endl;
}

TEST(TTKUnitTests, TestUniform) {
    Tensor2<double, 3> F = ttk::uniform<Tensor2<double, 3>>();
    std::cout << "F random = " << F << std::endl;
}

TEST(TTKUnitTests, TestRandomOrthogonal) {
    ttk::UniformDistribution<double> dist;
    Tensor2<double, 3> R = ttk::random_orthogonal<double, ttk::CARTESIAN>(dist);
    Tensor2<double, 3> Ihope = ttk::dot(R, ttk::transpose(R));
    Tensor2<double, 3> I = ttk::identity<Tensor2<double, 3>>();
    check_equal(Ihope, I);
    // assert this isn't a reflection
    double det = ttk::det(R);
    GTEST_ASSERT_GT(det, 0.0);
}

TEST(TTKUnitTests, TestEigen3x3) {
    Tensor2<double, 3> F = uniform_with_bounds(1e-8, 10.0);
    SymmetricTensor2<double, 3> C = ttk::tdot(F);
    auto [evals, evecs] = ttk::eigen(C);
    std::cout << "evals = " << evals << std::endl;
    Tensor2<double, 3> Diag;
    Diag.fill(0.0);
    for (int i = 0; i < 3; ++i) {
        Diag(i, i) = evals(i);
    }
    SymmetricTensor2<double, 3> Check = ttk::from_eigen(evals, evecs);
    check_equal(C, Check);

    Tensor2<double, 3> I(ttk::identity<Tensor2<double, 3>>());
    Tensor2<double, 3> CheckEvecs = ttk::dot(evecs, ttk::transpose(evecs));
    check_equal(I, CheckEvecs);
}

TEST(TTKUnitTests, TestEigen3x3Degenerate) {
    SymmetricTensor2<double, 3> C = 0.5 * ttk::identity<SymmetricTensor2<double, 3>>();
    auto [evals, evecs] = ttk::eigen(C);
    SymmetricTensor2<double, 3> Check = ttk::from_eigen(evals, evecs);
    check_equal(C, Check);
    Tensor2<double, 3> I(ttk::identity<Tensor2<double, 3>>());
    Tensor2<double, 3> CheckEvecs = ttk::dot(evecs, ttk::transpose(evecs));
    check_equal(I, CheckEvecs);
}

TEST(TTKUnitTests, TestLog3x3ScaledIdentity) {
    SymmetricTensor2<double, 3> C = 1.2 * ttk::identity<SymmetricTensor2<double, 3>>();
    SymmetricTensor2<double, 3> logC = ttk::log(C);
    SymmetricTensor2<double, 3> Check = ttk::log(1.2) * ttk::identity<SymmetricTensor2<double, 3>>();
    check_equal(Check, logC);
}

TEST(TTKUnitTests, TestLog3x3DoubleEigs) {
    ttk::UniformDistribution<double> dist;     
    SymmetricTensor2<double, 3> C(0.0);
    Tensor2<double, 3> R(ttk::random_orthogonal<double, 0>(dist));
    C(0, 0) = 2.0;
    C(1, 1) = 0.5;
    C(2, 2) = 2.0;
    C = ttk::symmetric(ttk::dot(R, ttk::dot(C, ttk::transpose(R))));
    SymmetricTensor2<double, 3> logC = ttk::log(C);
    SymmetricTensor2<double, 3> Check(0.0);
    Check(0, 0) = ttk::log(2.0);
    Check(1, 1) = ttk::log(0.5);
    Check(2, 2) = ttk::log(2.0);
    Check = ttk::symmetric(ttk::dot(R, ttk::dot(Check, ttk::transpose(R))));
    check_equal(Check, logC);
}

TEST(TTKUnitTests, TestSqrt3x3) {
    Tensor2<double, 3> F = uniform_with_bounds(1e-8, 10.0);
    SymmetricTensor2<double, 3> C = ttk::tdot(F);
    SymmetricTensor2<double, 3> U = ttk::sqrt(C);
    SymmetricTensor2<double, 3> Check = ttk::symmetric(ttk::dot(U, U));
    check_equal(Check, C);
}

TEST(TTKUnitTests, TestSqrt3x3ScaledIdentity) {
    SymmetricTensor2<double, 3> C = 1.2 * ttk::identity<SymmetricTensor2<double, 3>>();
    SymmetricTensor2<double, 3> sqrtC = ttk::sqrt(C);
    SymmetricTensor2<double, 3> Check = ttk::sqrt(1.2) * ttk::identity<SymmetricTensor2<double, 3>>();
    check_equal(Check, sqrtC);
}

TEST(TTKUnitTests, TestSqrt3x3DoubleEigs) {
    ttk::UniformDistribution<double> dist;     
    SymmetricTensor2<double, 3> C(0.0);
    Tensor2<double, 3> R(ttk::random_orthogonal<double, 0>(dist));
    C(0, 0) = 2.0;
    C(1, 1) = 0.5;
    C(2, 2) = 2.0;
    C = ttk::symmetric(ttk::dot(R, ttk::dot(C, ttk::transpose(R))));
    SymmetricTensor2<double, 3> sqrtC = ttk::sqrt(C);
    SymmetricTensor2<double, 3> Check(0.0);
    Check(0, 0) = ttk::sqrt(2.0);
    Check(1, 1) = ttk::sqrt(0.5);
    Check(2, 2) = ttk::sqrt(2.0);
    Check = ttk::symmetric(ttk::dot(R, ttk::dot(Check, ttk::transpose(R))));
    check_equal(Check, sqrtC);
}

TEST(TTKUnitTests, TestExp3x3ZeroIsIdentity) {
    SymmetricTensor2<double, 3> C(0.0);
    SymmetricTensor2<double, 3> expC = ttk::exp(C);
    SymmetricTensor2<double, 3> Check = ttk::identity<SymmetricTensor2<double, 3>>();
    check_equal(expC, Check);
}

TEST(TTKUnitTests, TestExp3x3ScaledIdentity) {
    SymmetricTensor2<double, 3> C = 1.2 * ttk::identity<SymmetricTensor2<double, 3>>();
    SymmetricTensor2<double, 3> expC = ttk::exp(C);
    SymmetricTensor2<double, 3> Check = ttk::exp(1.2) * ttk::identity<SymmetricTensor2<double, 3>>();
    check_equal(Check, expC);
}

TEST(TTKUnitTests, TestExp3x3DoubleEigs) {
    ttk::UniformDistribution<double> dist;     
    SymmetricTensor2<double, 3> C(0.0);
    Tensor2<double, 3> R(ttk::random_orthogonal<double, 0>(dist));
    C(0, 0) = 2.0;
    C(1, 1) = 0.5;
    C(2, 2) = 2.0;
    C = ttk::symmetric(ttk::dot(R, ttk::dot(C, ttk::transpose(R))));
    SymmetricTensor2<double, 3> expC = ttk::exp(C);
    SymmetricTensor2<double, 3> Check(0.0);
    Check(0, 0) = ttk::exp(2.0);
    Check(1, 1) = ttk::exp(0.5);
    Check(2, 2) = ttk::exp(2.0);
    Check = ttk::symmetric(ttk::dot(R, ttk::dot(Check, ttk::transpose(R))));
    check_equal(Check, expC);
}

TEST(TTKUnitTests, TestPow3x3ScaledIdentity) {
    SymmetricTensor2<double, 3> C = 1.2 * ttk::identity<SymmetricTensor2<double, 3>>();
    double m = 3.0;
    SymmetricTensor2<double, 3> powC = ttk::pow(C, m);
    SymmetricTensor2<double, 3> Check = ttk::pow(1.2, m) * ttk::identity<SymmetricTensor2<double, 3>>();
    check_equal(Check, powC);
}

TEST(TTKUnitTests, TestPow3x3DoubleEigs) {
    ttk::UniformDistribution<double> dist;     
    SymmetricTensor2<double, 3> C(0.0);
    double m = 0.25;
    Tensor2<double, 3> R(ttk::random_orthogonal<double, 0>(dist));
    C(0, 0) = 2.0;
    C(1, 1) = 0.5;
    C(2, 2) = 2.0;
    C = ttk::symmetric(ttk::dot(R, ttk::dot(C, ttk::transpose(R))));
    SymmetricTensor2<double, 3> powC = ttk::pow(C, m);
    SymmetricTensor2<double, 3> Check(0.0);
    Check(0, 0) = ttk::pow(2.0, m);
    Check(1, 1) = ttk::pow(0.5, m);
    Check(2, 2) = ttk::pow(2.0, m);
    Check = ttk::symmetric(ttk::dot(R, ttk::dot(Check, ttk::transpose(R))));
    check_equal(Check, powC);
}

} // end ttk_cartesian_tests
