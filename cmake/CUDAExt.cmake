if(TENSOR_TOOLKIT_ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "Enabling CUDA support")
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Reasonable defaults
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    # Let users override architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80)
    endif()
endif()
