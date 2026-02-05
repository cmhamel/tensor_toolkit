#pragma once
#include <gtest/gtest.h>
#include <tensor_toolkit.hpp>

namespace ttk_continuum_tests {

using MaterialVector = ttk::MaterialVector<double, ttk::CARTESIAN, 3>;
using SpatialVector = ttk::SpatialVector<double, ttk::CARTESIAN, 3>;
using MaterialTensor2 = ttk::MaterialTensor2<double, ttk::CARTESIAN, 3>;
using SpatialTensor2 = ttk::SpatialTensor2<double, ttk::CARTESIAN, 3>;
using TwoPointTensor2 = ttk::TwoPointTensor2<double, ttk::CARTESIAN, 3>;

template<typename TensorType>
TTK_FUNCTION
void test_identity_1(TensorType v) {
    EXPECT_DOUBLE_EQ(v(0), 1.0);
    EXPECT_DOUBLE_EQ(v(1), 1.0);
    EXPECT_DOUBLE_EQ(v(2), 1.0);
}

template<typename TensorType>
TTK_FUNCTION
void test_identity_2(TensorType I) {
    EXPECT_DOUBLE_EQ(I(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(I(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(I(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(I(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(I(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(I(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(I(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(I(2, 1), 0.0);
    EXPECT_DOUBLE_EQ(I(2, 2), 1.0);
}

TEST(TTKContinuumUnitTests, TestIdentities) {
    MaterialVector vm(ttk::identity<MaterialVector>());
    test_identity_1(vm);
    SpatialVector vs(ttk::identity<SpatialVector>());
    test_identity_1(vs);
    MaterialTensor2 Im(ttk::identity<MaterialTensor2>());
    test_identity_2(Im);
    SpatialTensor2 Is(ttk::identity<SpatialTensor2>());
    test_identity_2(Is);
    TwoPointTensor2 It(ttk::identity<TwoPointTensor2>());
    test_identity_2(It);
    std::cout << "I = " << It << std::endl;
}

TEST(TTKContinuumUnitTests, TestCross) {
    MaterialVector am({1., 2., 3.});
    MaterialVector bm({4., 5., 6.});
    MaterialVector cm = ttk::cross(am, bm);
    EXPECT_DOUBLE_EQ(cm(0), -3.0);
    EXPECT_DOUBLE_EQ(cm(1),  6.0);
    EXPECT_DOUBLE_EQ(cm(2), -3.0);

    SpatialVector as({1., 2., 3.});
    SpatialVector bs({4., 5., 6.});
    SpatialVector cs = ttk::cross(as, bs);
    EXPECT_DOUBLE_EQ(cs(0), -3.0);
    EXPECT_DOUBLE_EQ(cs(1),  6.0);
    EXPECT_DOUBLE_EQ(cs(2), -3.0);

    // this one will fail to compile
    // ttk::cross(am, cs);
    // // this one will also fail to compile
    // ttk::cross(cs, am);
}



} // end namespace