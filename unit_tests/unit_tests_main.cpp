#include <gtest/gtest.h>
#include "cartesian_unit_tests.hpp"
#include "continuum_unit_tests.hpp"
#include "kokkos_unit_tests.hpp"
#include "material_model_tests.hpp"
#include "simple_function_unit_tests.hpp"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
