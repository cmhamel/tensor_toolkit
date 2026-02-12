cmake_minimum_required(VERSION 3.16)
project(tensor_toolkit CXX)

# ---------- Options ----------
option(TENSOR_TOOLKIT_ENABLE_CUDA "Enable CUDA support" OFF)
option(TENSOR_TOOLKIT_ENABLE_ENZYME "Enable Enzyme support" OFF)
option(TENSOR_TOOLKIT_ENABLE_HIP  "Enable HIP/ROCm support" OFF)
option(TENSOR_TOOLKIT_ENABLE_KOKKOS "Enable Kokkos support" OFF)
# option(TENSOR_TOOLKIT_ENABLE_KOKKOS_TESTS "Enable Kokkos testing support" OFF)
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
target_compile_options(tensor_toolkit INTERFACE
    -Wall
    -Wunused-variable
)
target_compile_options(tensor_toolkit INTERFACE
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-fstack-usage>
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-Wframe-larger-than=512>
    $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:Clang>>:-fno-omit-frame-pointer>
)
target_compile_options(tensor_toolkit INTERFACE
    $<$<CXX_COMPILER_ID:GNU>:-fpermissive>
)

# setup executable first to make the ext cmake files simple
if(TENSOR_TOOLKIT_ENABLE_TESTS)
    add_executable(ttk_unit_tests
        unit_tests/unit_tests_main.cpp
    )
endif()

# Extensions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CUDAExt)
include(EnzymeExt)
include(HIPExt)
include(KokkosExt)

# ---------------- Tests ----------------
if(TENSOR_TOOLKIT_ENABLE_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    target_link_libraries(ttk_unit_tests
        PRIVATE
            tensor_toolkit
            GTest::gtest_main
    )
    include(GoogleTest)
    add_test(
        NAME ttk_unit_tests
        COMMAND ttk_unit_tests
    )
endif()

# Installation
include(GNUInstallDirs)
# Install the interface library target
install(
    TARGETS tensor_toolkit
    EXPORT tensor_toolkitTargets
)

# Install headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ttk
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(
    EXPORT tensor_toolkitTargets
    FILE tensor_toolkitTargets.cmake
    NAMESPACE tensor_toolkit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor_toolkit
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tensor_toolkitConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tensor_toolkitConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/tensor_toolkitConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor_toolkit
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tensor_toolkitConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tensor_toolkitConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor_toolkit
)
