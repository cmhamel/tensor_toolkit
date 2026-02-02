if(TENSOR_TOOLKIT_ENABLE_HIP)
    message(STATUS "Enabling HIP/ROCm support")

    find_package(HIP REQUIRED)

    # Make HIP behave like C++
    set(HIP_STANDARD 20)
    set(HIP_STANDARD_REQUIRED ON)
    # set(CMAKE_CXX_COMPILER hipcc)(

    if(TENSOR_TOOLKIT_ENABLE_TESTS)
        # if(TENSOR_TOOLKIT_ENABLE_HIP)
        #     set(Kokkos_ENABLE_HIP ON)
        # endif()
    endif()
endif()
