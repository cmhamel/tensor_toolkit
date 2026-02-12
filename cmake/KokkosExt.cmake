if(TENSOR_TOOLKIT_ENABLE_KOKKOS)
    find_package(Kokkos REQUIRED)
    if(TENSOR_TOOLKIT_ENABLE_CUDA)
        set(Kokkos_ENABLE_CUDA ON)
    endif()
    if(TENSOR_TOOLKIT_ENABLE_HIP)
        set(Kokkos_ENABLE_HIP ON)
    endif()

    target_link_libraries(tensor_toolkit
        PRIVATE INTERFACE
            Kokkos::kokkos
    )
    if(TENSOR_TOOLKIT_ENABLE_TESTS)
        target_link_libraries(ttk_unit_tests
            PRIVATE INTERFACE
                Kokkos::kokkos
        )
    endif()
endif()
