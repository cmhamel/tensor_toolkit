#pragma once
#include <cmath>
#include <gtest/gtest.h>
#include <ttk/tensor_toolkit.hpp>

namespace ttk_simple_function_tests {

// TODO add GPU tests maybe?

TEST(TTKSimpleFunctionTests, cbrt) {
    double val = 1.5;
    EXPECT_NEAR(ttk::cbrt(val), std::cbrt(val), 1e-12);
}

TEST(TTKSimpleFunctionTests, pow) {
    double val = 2.5;
    double exp = 3.5;
    EXPECT_NEAR(ttk::pow(val, exp), std::pow(val, exp), 1e-12);
}

TEST(TTKSimpleFunctionTests, sqrt) {
    double val = 1.5;
    EXPECT_NEAR(ttk::sqrt(val), std::sqrt(val), 1e-12);
}

} // end namespace
