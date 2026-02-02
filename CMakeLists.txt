cmake_minimum_required(VERSION 3.5.0)
project(tensor_toolkit CXX)

# ---------- Options ----------
option(TENSOR_TOOLKIT_ENABLE_CUDA "Enable CUDA support" OFF)
option(TENSOR_TOOLKIT_ENABLE_ENZYME "Enable Enzyme support" OFF)
option(TENSOR_TOOLKIT_ENABLE_HIP  "Enable HIP/ROCm support" OFF)
option(TENSOR_TOOLKIT_ENABLE_KOKKOS_TESTS "Enable Kokkos testing support" OFF)
option(TENSOR_TOOLKIT_ENABLE_TESTS "Build unit tests" ON)

# ---------- C++ standard ----------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Library ----------
add_library(tensor_toolkit INTERFACE)
target_include_directories(tensor_toolkit
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
file(GLOB TTK_SOURCES "src/*.cpp")

# ---------------- Tests ----------------
if(TENSOR_TOOLKIT_ENABLE_TESTS)
    enable_testing()
    find_package(GTest)
    add_executable(ttk_unit_tests
        unit_tests/unit_tests_main.cpp
    )
    target_link_libraries(ttk_unit_tests
        PRIVATE
            tensor_toolkit
            GTest::gtest_main
    )
    include(GoogleTest)
endif()

# Extensions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CUDAExt)
include(EnzymeExt)
include(HIPExt)
include(KokkosExt)
