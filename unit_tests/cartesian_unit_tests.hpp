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
    std::cout << "I = " << std::endl;
    std::cout << I << std::endl;
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

} // end ttk_cartesian_tests
