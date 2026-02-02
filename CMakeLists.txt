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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "GCC detected: adding -fpermissive flag")
    # fails on compiling identity method without this
    add_compile_options(-fpermissive)
    
    # Option 2: per target
    # target_compile_options(tensor_toolkit PRIVATE -fpermissive)
endif()

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

    # just so we can pass on CI with windows
    if(WIN32)
        # Fetch GoogleTest if not found
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        )
        # Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

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
    add_test(
        NAME ttk_unit_tests
        COMMAND ttk_unit_tests
    )
endif()

# Extensions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CUDAExt)
include(EnzymeExt)
include(HIPExt)
include(KokkosExt)

# Installation
include(GNUInstallDirs)
# Install the interface library target
install(
    TARGETS tensor_toolkit
    EXPORT tensor_toolkitTargets
)

# Install headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(
    EXPORT tensor_toolkitTargets
    FILE tensor_toolkitTargets.cmake
    NAMESPACE tensor_toolkit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor_toolkit
)

# @PACKAGE_INIT@

# include("${CMAKE_CURRENT_LIST_DIR}/tensor_toolkitTargets.cmake")

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
